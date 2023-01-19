from __future__ import annotations
import copy

from typing import Callable, Optional, Union

from pypardiso import spsolve
from constitutive_laws import (
    DifferentiatedDarcyLaw,
    PowerLawPermeability,
    DissolutionReaction,
    PrecipitationPorosity,
)
from common_models import DataSaving, Geometry, SolutionStrategyMixin
import porepy as pp
import numpy as np


class SolutionStrategyChemistry(pp.SolutionStrategy):
    """Reactive transport of one dissolved component and one precipitate."""

    pressure: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Pressure ad variable."""
    pressure_variable: str
    """Name of the pressure variable."""
    equation_system: pp.EquationSystem
    bc_type_solute: Callable[[pp.Grid], pp.BoundaryCondition]

    def __init__(self, params: dict):
        super().__init__(params)

        # Names of the primary variables.
        self.component_a_variable = "A"
        self.component_b_variable = "B"
        # Discretization keyword for chemistry.
        self.chemistry_keyword = "chemistry_parameters"

    def initial_condition(self):
        """Initial condition for all primary variables and the parameter representing the
        secondary flux variable (needs to be present for first call to upwind discretization).

        """
        super().initial_condition()
        for sd, data in self.mdg.subdomains(return_data=True):
            state = {
                self.component_a_variable: self.component.solute_fraction()
                * np.ones(sd.num_cells),
                self.component_b_variable: self.component.precipitate_fraction()
                * np.ones(sd.num_cells),
                self.pressure_variable: self.fluid.pressure() * np.ones(sd.num_cells),
            }
            data[pp.STATE].update(state)
            data[pp.STATE][pp.ITERATE].update(copy.deepcopy(state))
            pp.initialize_data(
                sd, data, self.chemistry_keyword, {"darcy_flux": np.zeros(sd.num_faces)}
            )

    def before_nonlinear_loop(self) -> None:
        # Do a pure Darcy solve before the chemistry loop
        a, b = self.equation_system.assemble_subsystem(
            ["mass_balance_equation"], variables=[self.pressure_variable]
        )
        x = spsolve(a, b)
        self.equation_system.set_variable_values(
            values=x,
            variables=[self.pressure_variable],
            to_state=True,
            to_iterate=True,
            additive=False,
        )
        super().before_nonlinear_loop()

    def before_nonlinear_iteration(self):
        """Evaluate Darcy flux (super) and copy to the enthalpy flux keyword, to be used
        in upstream weighting.

        """
        super().before_nonlinear_iteration()
        for _, data in self.mdg.subdomains(return_data=True) + self.mdg.interfaces(
            return_data=True
        ):
            vals = data[pp.PARAMETERS][self.mobility_keyword]["darcy_flux"]
            data[pp.PARAMETERS][self.chemistry_keyword].update({"darcy_flux": vals})

        # TODO: Targeted rediscretization.
        self.discretize()

    def set_discretization_parameters(self) -> None:
        super().set_discretization_parameters()
        for sd, data in self.mdg.subdomains(return_data=True):
            pp.initialize_data(
                sd, data, self.chemistry_keyword, {"bc": self.bc_type_solute(sd)}
            )


class VariablesChemistry:
    component_a_variable: str
    component_b_variable: str
    equation_system: pp.EquationSystem
    mdg: pp.MixedDimensionalGrid

    def create_variables(self) -> None:
        """Assign primary variables to the single subdomain of the mixed-dimensional grid."""
        super().create_variables()
        # First for the subdomains
        self.equation_system.create_variables(
            self.component_a_variable,
            subdomains=self.mdg.subdomains(),
        )
        self.equation_system.create_variables(
            self.component_b_variable,
            subdomains=self.mdg.subdomains(),
        )

    def component_b(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        return self.equation_system.md_variable(self.component_b_variable, subdomains)

    def component_a(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        return self.equation_system.md_variable(self.component_a_variable, subdomains)


class ComponentConstants(pp.MaterialConstants):
    def __init__(self, constants: Optional[dict[str, pp.number]] = None):
        # Default values, sorted alphabetically
        default_constants: dict[str, pp.number] = {
            "reaction_rate": 1e-0,
            "precipitate_fraction": 4e-2,
            "solute_fraction": 1e-1,
            "equilibrium_constant": 1e-1,
        }
        if constants is not None:
            default_constants.update(constants)
        super().__init__(default_constants)

    def equilibrium_constant(self) -> pp.number:
        """Equilibrium constant.

        Returns:
            The equilibrium constant.

        """
        return self.convert_units(self.constants["equilibrium_constant"], "-")

    def precipitate_fraction(self) -> pp.number:
        """Fraction of precipitate.

        Returns:
            The precipitate fraction.

        """
        return self.convert_units(self.constants["precipitate_fraction"], "-")

    def reaction_rate(self) -> pp.number:
        """Reference rate r_0 of the reaction term.

        Returns:
            The reference reaction rate.

        TODO: Check units.

        """
        return self.convert_units(self.constants["reaction_rate"], "s^-1")

    def solute_fraction(self) -> pp.number:
        """Fraction of solute.

        Returns:
            The solute fraction.

        """
        return self.convert_units(self.constants["solute_fraction"], "-")


class ChemistryEquations(pp.BalanceEquation):
    porosity: Callable[[list[pp.Grid]], pp.ad.Operator]
    component_a: Callable[[list[pp.Grid]], pp.ad.Operator]
    component_b: Callable[[list[pp.Grid]], pp.ad.Operator]
    reference_reaction_rate: Callable[[list[pp.Grid]], pp.ad.Operator]
    solute_source: Callable[[list[pp.Grid]], pp.ad.Operator]
    solute_sink: Callable[[list[pp.Grid]], pp.ad.Operator]
    equilibrium_constant: Callable[[list[pp.Grid]], pp.ad.Operator]
    reaction_term: Callable[[list[pp.Grid]], pp.ad.Operator]
    bc_values_solute: Callable[[list[pp.Grid]], pp.ad.Operator]
    advective_flux: Callable
    mobility: Callable[[list[pp.Grid]], pp.ad.Operator]
    chemistry_keyword: str

    def set_equations(self):
        """Set the equations for the mass balance problem.

        A mass balance equation is set for all subdomains and a Darcy-type flux relation
        is set for all interfaces of codimension one.

        """
        super().set_equations()
        subdomains = self.mdg.subdomains()
        a_eq = self.component_a_balance_equation(subdomains)
        precipitate_eq = self.precipitate_balance_equation(subdomains)
        self.equation_system.set_equation(a_eq, subdomains, {"cells": 1})
        self.equation_system.set_equation(precipitate_eq, subdomains, {"cells": 1})

    def precipitate_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Precipitate mass balance equation.


        Parameters:
            subdomains: Subdomains on which the equation is defined.

        Returns:
            eq: The equation on AD form.

        """
        # Mass matrix to integrate over cell
        accumulation = self.volume_integral(
            self.component_b(subdomains),
            subdomains,
            dim=1,
        )

        reaction_term = self.volume_integral(
            self.reaction_term(subdomains), subdomains, 1
        )
        size = sum([sd.num_faces for sd in subdomains])
        flux_term = pp.wrap_as_ad_array(0, size)
        eq = self.balance_equation(
            subdomains, accumulation, flux_term, reaction_term, dim=1
        )
        eq.set_name("Precipitate balance equation")
        return eq

    def component_a_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Advective transport equations.


        Parameters:
            subdomains: Subdomains on which the equation is defined.

        Returns:
            eq: The equation on AD form.

        """
        source = self.solute_source(subdomains)
        sink = self.solute_sink(subdomains)

        accumulation = self.volume_integral(
            self.porosity(subdomains) * self.component_a(subdomains),
            subdomains,
            dim=1,
        )

        advection = self.solute_flux(subdomains)

        reaction_term = self.volume_integral(
            self.reaction_term(subdomains), subdomains, 1
        )

        source_terms = source + sink * self.component_a(subdomains) - reaction_term
        eq = self.balance_equation(
            subdomains, accumulation, advection, source_terms, dim=1
        )
        eq.set_name("component_a_balance_equation")
        return eq

    def solute_flux(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Solute flux.

        Parameters:
            subdomains: Subdomains on which the equation is defined.

        Returns:
            eq: The equation on AD form.

        """
        flux = self.advective_flux(
            subdomains,
            self.component_a(subdomains) * self.mobility(subdomains),
            pp.ad.UpwindAd(self.chemistry_keyword, subdomains),
            self.bc_values_solute(subdomains),
            None,
        )
        return flux


class BoundaryConditionsChemistry:
    """Boundary conditions for the solute transport equation."""

    domain_boundary_sides: Callable[[pp.Grid], pp.bounding_box.DomainSides]
    component: ComponentConstants

    def bc_type_solute(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Dirichlet conditions on all external boundaries.

        Parameters:
            sd: Subdomain grid on which to define boundary conditions.

        Returns:
            Boundary condition object.

        """
        # Define boundary faces.
        boundary_faces = self.domain_boundary_sides(sd).all_bf
        # Define boundary condition on all boundary faces.
        return pp.BoundaryCondition(sd, boundary_faces, "dir")

    def bc_values_solute(self, subdomains: list[pp.Grid]) -> pp.ad.Array:
        """Boundary condition values for the mobility times density.

        Units:

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Array with boundary values for the mobility.

        """
        # List for all subdomains
        bc_values: list[np.ndarray] = []

        # Loop over subdomains to collect boundary values
        for sd in subdomains:
            # Get density and viscosity values on boundary faces applying trace to
            # interior values.
            # Define boundary faces.
            boundary_faces = self.domain_boundary_sides(sd).all_bf
            # Append to list of boundary values
            vals = np.zeros(sd.num_faces)
            vals[boundary_faces] = self.component.solute_fraction() / 2
            bc_values.append(vals)
        # Concatenate to single array and wrap as ad.Array
        # We have forced the type of bc_values_array to be an ad.Array, but mypy does
        # not recognize this. We therefore ignore the typing error.
        bc_values_array: pp.ad.Array = pp.wrap_as_ad_array(  # type: ignore
            np.hstack(bc_values), name="bc_values_a"
        )
        return bc_values_array


class ConstitutiveLawsChemistry(
    DissolutionReaction, PrecipitationPorosity, PowerLawPermeability
):
    """Class for constitutive laws for the chemistry problem."""

    def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Permeability tensor.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Permeability tensor.

        """
        return self.power_law_permeability(subdomains)


class Chemistry(
    DataSaving,
    Geometry,
    ConstitutiveLawsChemistry,
    BoundaryConditionsChemistry,
    DifferentiatedDarcyLaw,
    VariablesChemistry,
    ChemistryEquations,
    SolutionStrategyChemistry,
    SolutionStrategyMixin,
    pp.fluid_mass_balance.SinglePhaseFlow,
):
    pass
