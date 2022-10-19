from typing import Dict, List

import numpy as np
import porepy as pp

from models import BiotNonlinearTpfa, Chemistry, NonlinearIncompressibleFlow
from run_chemistry import ChemistryParameters
from run_verification import SquareRootPermeability


class EquilibriumChemistryParameters(ChemistryParameters):
    def _set_chemistry_parameters(self) -> None:
        super()._set_chemistry_parameters()
        comp = "b"
        for sd, data in self.mdg:
            parameters = {
                "mass_weight": np.ones(sd.num_cells),
                "bc": getattr(self, "_bc_type_" + comp)(sd),
                "bc_values": getattr(self, "_bc_values_" + comp)(sd),
                "source": getattr(self, "_source_" + comp)(sd),
                "stoichiometric_coefficient": getattr(
                    self, "_stoichiometric_coefficient_" + comp
                )(sd),
            }
            pp.initialize_data(
                sd,
                data,
                getattr(self, comp + "_parameter_key"),
                parameters,
            )

    def _bc_type_b(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Dirichlet conditions on all external boundaries."""
        # Define boundary regions
        all_bf, *_ = self._domain_boundary_sides(sd)
        # Define boundary condition on faces
        return pp.BoundaryCondition(sd, all_bf, "dir")

    def _bc_values_b(self, sd: pp.Grid) -> np.ndarray:
        """Homogeneous boundary values.

        Units:
            Dirichlet conditions: Pa = kg / m^1 / s^2
            Neumann conditions: m^3 / s
        """
        return np.zeros(sd.num_faces)

    def _stoichiometric_coefficient_b(self, sd: pp.Grid) -> int:
        """Stoichiometric coefficient for the chemical reaction

        Args:
            sd: pp.Grid

        Returns:
            np.ndarray: (g.num_cells)

        """
        return self._beta * np.ones(sd.num_cells)

    def _source_b(self, sd: pp.Grid) -> np.ndarray:
        """Homogeneous values except at the domain center."""

        return self._domain_center_source(sd, 0.55) * sd.cell_volumes

    def _initial_b(self, sd: pp.Grid) -> np.ndarray:
        """Homogeneous initial values.

        Args:
            sd: pp.Grid of the subdomain.

        Returns:
            vals: np.ndarray of cell-wise concentration values
        """
        return 0.5 * np.ones(sd.num_cells)


class ShearDilation(BiotNonlinearTpfa):
    def _bc_values_mechanics(self, sd: pp.Grid) -> np.ndarray:
        vals = np.zeros((self.nd, sd.num_faces))
        faces = sd.face_centers[self.nd - 1] > 1 - 1e-7
        vals[self.nd - 1, faces] = -1e-3
        vals[self.nd - 2, faces] = 2e-3
        return vals.ravel(order="F")

    def _dilation_angle(self, sd):
        return 0.2 * np.ones(sd.num_cells)

    def _friction_coefficient(self, sd):
        return 5e-1 * np.ones(sd.num_cells)

    def _source_scalar(self, sd: pp.Grid) -> np.ndarray:
        """Homogeneous Dirichlet values except at left end of fracture."""
        vals = np.zeros(sd.num_cells)
        if sd.dim == self.nd - 1:
            pt = np.zeros(3)
            ind = sd.closest_cell(pt)
            val = 0.01
            if sd.dim == 1:
                val = 8e-3
            vals[ind] = val
        return vals


class SquarePermeability2d(SquareRootPermeability):
    def _source(self, sd: pp.Grid) -> np.ndarray:
        """Source term.

        Units: m^3 / s
        """
        x = sd.cell_centers[0]
        y = sd.cell_centers[1]
        val = (
            -2 * x * (1 - x) * (x**2 + 0.1)
            + 2 * x * (-x * y * (1 - y) + y * (1 - x) * (1 - y))
            - 2 * y * (1 - y) * (x**2 + 0.1)
        )
        return val * sd.cell_volumes

    def p_analytical(self, sd=None):
        if sd is None:
            sd = self.gb.grids_of_dimension(self.gb.dim_max())[0]
        x = sd.cell_centers[0]
        y = sd.cell_centers[1]
        return x * y * (1 - x) * (1 - y)

    def _permeability_function(self, pressure: pp.ad.Variable):
        val = pressure**2 + 0.1
        return val

    def _permeability_function_ad(self, pressure: pp.ad.Variable):
        val = pressure**2 + 0.1
        return val


class EquilibriumChemistry(Chemistry):
    def __init__(self, params: Dict):
        super().__init__(params)
        self.component_b_variable = "B"
        self.b_parameter_key = "chemistry_parameters_b"
        # Stoichiometric constant
        self._beta = self.params.get("beta", 1)
        self._equilibrium_constant = self.params.get("equilibrium_constant", 0.25)

    def _initial_condition(self):
        """Initial condition for all primary variables and the parameter representing the
        secondary flux variable (needs to be present for first call to upwind discretization).
        """
        # Pressure, a, b and flux for a-parameter
        super()._initial_condition()
        # add b and flux for b-parameter
        for g, d in self.mdg:
            state = {self.component_b_variable: self._initial_b(g)}
            d[pp.STATE].update(state)
            d[pp.STATE][pp.ITERATE].update(state.copy())
            pp.initialize_data(
                g, d, self.b_parameter_key, {"darcy_flux": np.zeros(g.num_faces)}
            )

    def _assign_variables(self) -> None:
        """
        Assign primary variables to the nodes and edges of the grid bucket.
        """
        super()._assign_variables()
        # First for the nodes
        for g, d in self.mdg:
            d[pp.PRIMARY_VARIABLES].update({self.component_b_variable: {"cells": 1}})

    def _create_ad_variables(self) -> None:
        """Create the merged variables for potential and mortar flux"""
        super()._create_ad_variables()
        grid_list = [g for g, _ in self.mdg.nodes()]
        self._ad.component_b = self._eq_manager.merge_variables(
            [(g, self.component_b_variable) for g in grid_list]
        )

    def _assign_equations(self) -> None:
        """Define equations through discretizations.

        Assigns a Laplace/Darcy problem discretized using Mpfa on all subdomains with
        Neumann conditions on all internal boundaries. On edges of co-dimension one,
        interface fluxes are related to higher- and lower-dimensional pressures using
        the RobinCoupling.

        Gravity is included, but may be set to 0 through assignment of the vector_source
        parameter.
        """
        NonlinearIncompressibleFlow._assign_equations(self)

        subdomain_transport_equation_a: pp.ad.Operator = (
            self._subdomain_transport_equation(self.grid_list, component="a")
        )
        subdomain_transport_equation_b: pp.ad.Operator = (
            self._subdomain_transport_equation(self.grid_list, component="b")
        )
        equilibrium_equation: pp.ad.Operator = self._equilibrium_equation(
            self.grid_list
        )
        # Assign equations to manager
        self._eq_manager.name_and_assign_equations(
            {
                "subdomain_transport_a": subdomain_transport_equation_a,
                "subdomain_transport_b": subdomain_transport_equation_b,
                "equilibrium": equilibrium_equation,
            },
        )

    def _equilibrium_equation(self, subdomains: List[pp.Grid]) -> pp.ad.Operator:
        """Mass balance equation for slightly compressible flow in a deformable medium.


        Parameters
        ----------
        subdomains : List[pp.Grid]
            Subdomains on which the equation is defined.

        Returns
        -------
        eq : pp.ad.Operator
            The equation on AD form.

        """
        equilibrium_constant = pp.ad.ParameterArray(
            param_keyword=self.c_parameter_key,
            array_keyword="equilibrium_constant",
            grids=subdomains,
        )
        log = pp.ad.Function(pp.ad.functions.log, "logarithm")
        eq: pp.ad.Operator = (
            log(equilibrium_constant)
            - self._alpha * log(self._ad.component_a)
            - self._beta * log(self._ad.component_b)
        )

        return eq

    def before_newton_iteration(self):
        super().before_newton_iteration()
        self._ad.dummy_upwind_for_discretization_b.discretize(self.mdg)

    def compute_fluxes(self) -> None:
        super().compute_fluxes()
        # Copy to b parameters
        for _, d in self.mdg:
            d[pp.PARAMETERS][self.b_parameter_key]["darcy_flux"] = d[pp.PARAMETERS][
                self.a_parameter_key
            ]["darcy_flux"]

    def _export_names(self):
        return super()._export_names() + [self.component_b_variable]


class LinearPermeability2d:
    """Simple linear permeability for pure flow.

        K(p) = k0 + p

    """

    def _source(self, sd: pp.Grid):
        x = sd.cell_centers[0]
        y = sd.cell_centers[1]
        k0 = self.params["k0"]
        val = (
            -2 * x * (1 - x) * (x * y * (1 - x) * (1 - y) + k0)
            - 2 * y * (1 - y) * (x * y * (1 - x) * (1 - y) + k0)
            + (-x * y * (1 - x) + x * (1 - x) * (1 - y)) ** 2
            + (-x * y * (1 - y) + y * (1 - x) * (1 - y)) ** 2
        )
        return -val * sd.cell_volumes

    def p_analytical(self, sd=None):
        if sd is None:
            sd = self.mdg.subdomains(dim=self.mdg.dim_max())[0]
        x = sd.cell_centers[0]
        y = sd.cell_centers[1]
        return x * y * (1 - x) * (1 - y)

    def _permeability_function(self, pressure: pp.ad.Variable):
        k0 = self.params["k0"]
        val = pressure + k0
        return val

    def _permeability_function_ad(self, pressure: pp.ad.Variable):
        nc = pressure.val.size
        k0 = pp.ad.Ad_array(
            self.params["k0"] * np.ones(nc),
            sps.csr_matrix((nc, self.dof_manager.num_dofs()), dtype=float),
        )
        K = pressure + k0
        return K

