from typing import Any, Callable
import porepy as pp

from chemistry_model import ComponentConstants


class DifferentiatedDarcyLaw(pp.constitutive_laws.DarcysLaw):
    params: dict[str, Any]
    equation_system: pp.EquationSystem
    permeability: Callable[[list[pp.Grid]], pp.ad.Operator]
    permeability_argument: Callable[[list[pp.Grid]], pp.ad.Operator]

    def darcy_flux(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:

        base_discr = self.darcy_flux_discretization(subdomains)
        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(subdomains, [1])

        if self.params.get("use_linear_discretization", False):
            return super().darcy_flux(subdomains)
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        flux: pp.ad.Operator = (
            self.differentiable_tpfa(subdomains).flux()
            + (
                base_discr.bound_flux * self.bc_values_darcy(subdomains)
                + base_discr.bound_flux
                * projection.mortar_to_primary_int
                * self.interface_darcy_flux(interfaces)
            )
            + base_discr.vector_source
            * self.vector_source(subdomains, material="fluid")
        )
        return flux

    def differentiable_tpfa(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.DifferentiableFVAd:
        """Discretization object for the Darcy flux term.

        Parameters:
            subdomains: List of subdomains where the Darcy flux is defined.

        Returns:
            Discretization of the Darcy flux.

        """
        base_discr = self.darcy_flux_discretization(subdomains)
        p = self.pressure(subdomains)
        f = getattr(self, "transmissibility", self.permeability)
        tpfa = pp.ad.DifferentiableFVAd(
            subdomains=subdomains,
            mdg=self.mdg,
            base_discr=base_discr,
            equation_system=self.equation_system,
            permeability_function=f,
            permeability_argument=self.permeability_argument(subdomains),
            potential=p,
            keyword=self.darcy_keyword,
        )
        return tpfa

    def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.MpfaAd:
        """Discretization object for the Darcy flux term.

        Parameters:
            subdomains: List of subdomains where the Darcy flux is defined.

        Returns:
            Discretization of the Darcy flux.

        """
        if self.params.get("use_tpfa", False):
            return pp.ad.TpfaAd(self.darcy_keyword, subdomains)
        else:
            return pp.ad.MpfaAd(self.darcy_keyword, subdomains)

    def pressure_trace(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        if self.params.get("use_linear_discretization", False):
            return super().pressure_trace(subdomains)

        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(subdomains, [1])
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        discr: pp.ad.MpfaAd = self.darcy_flux_discretization(subdomains)
        tpfa_diff = self.differentiable_tpfa(subdomains)

        p_primary = (
            discr.bound_pressure_cell * self.pressure(subdomains)
            + tpfa_diff.bound_pressure_face()
            * (projection.mortar_to_primary_int * self.interface_darcy_flux(interfaces))
            + tpfa_diff.bound_pressure_face() * self.bc_values_darcy(subdomains)
            + discr.vector_source * self.vector_source(subdomains, material="fluid")
        )
        return p_primary


class PowerLawPermeability:
    """Power law permeability-porosity relation."""

    params: dict[str, Any]
    """Dictionary of parameters. Should contain the key "permeability_exponent"."""
    porosity: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Porosity function."""
    solid: pp.SolidConstants
    """Solid constants."""

    def power_law_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        K_0 = pp.ad.Scalar(self.solid.permeability(), name="reference_permeability")
        phi_0 = pp.ad.Scalar(self.solid.porosity())
        eta = self.params.get("permeability_exponent", 3)
        phi = self.porosity(subdomains)

        perm = K_0 * (phi / phi_0) ** eta
        return perm


class PoromechanicsPermeability(
    PowerLawPermeability, pp.constitutive_laws.CubicLawPermeability
):
    """Permeability functions for poromechanics.

    Power law in matrix and cubic law in fractures.
    """

    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid."""
    nd: int
    """Number of dimensions."""

    def matrix_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Permeability in the matrix.

        Parameters:
            subdomains: List of subdomains. Should contain only one fracture grid.

        Returns:
            Cell-wise permeability values.

        """
        return self.power_law_permeability(subdomains)

    def transmissibility(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Transmissibility function.

        Parameters:
            subdomains: List of subdomains. Should contain only one fracture grid.

        Returns:
            Cell-wise transmissibility values.

        """
        return self.permeability(subdomains) * self.specific_volume(subdomains)

    def permeability_argument(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Permeability argument for the Darcy flux.

        Parameters:
            subdomains: List of subdomains where the Darcy flux is defined.

        Returns:
            Permeability argument.

        """
        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(subdomains, [1])
        # return self.interface_displacement(interfaces)
        return self.displacement([sd for sd in subdomains if sd.dim == self.nd])


class PrecipitationPorosity:
    solid: pp.SolidConstants
    component_b: Callable[[list[pp.Grid]], pp.ad.Operator]

    def porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Porosity, including the effect of solid precipitation.

        Parameters:
            subdomains: List of subdomain grids.

        Returns:
            porosity: pp.ad.Operator representing the porosity.
        """
        phi_0 = pp.ad.Scalar(self.solid.porosity())
        phi = phi_0 - self.component_b(subdomains) * phi_0
        return phi


class DissolutionReaction:
    """Constitutive laws for the chemistry model."""

    solid: pp.SolidConstants
    component: ComponentConstants
    porosity: Callable[[list[pp.Grid]], pp.ad.Operator]
    component_a: Callable[[list[pp.Grid]], pp.ad.Operator]

    def reference_reaction_rate(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Reference rate r_0 of the reaction term.

        Parameters:
            sd: pp.Grid representing a subdomain.

        Returns:
            r_0: array of cell-wise rates. Homogeneous values is the obvious choice.

        """
        r_0 = pp.ad.Scalar(self.component.reaction_rate(), name="reaction_rate")
        return r_0

    def reaction_term(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Reaction term.

        Parameters:
            subdomains: Subdomains on which the equation is defined.

        Returns:
            Reaction term on AD form.

        """
        r_0 = self.reference_reaction_rate(subdomains)
        equilibrium_constant_inv = self.equilibrium_constant(subdomains) ** -1
        rate = r_0 * (equilibrium_constant_inv * self.component_a(subdomains) - 1)
        return rate

    def equilibrium_constant(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Equilibrium constant.

        Parameters:
            subdomains: Subdomains on which the equation is defined.

        Returns:
            Equilibrium constant on AD form.

        """
        return pp.ad.Scalar(
            self.component.equilibrium_constant(), name="equilibrium_constant"
        )
