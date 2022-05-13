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
        for g, d in self.gb:
            parameters = {"mass_weight": np.ones(g.num_cells),
                          "bc": getattr(self, "_bc_type_" + comp)(g),
                          "bc_values": getattr(self, "_bc_values_" + comp)(g),
                          "source": getattr(self, "_source_" + comp)(g),
                          "stoichiometric_coefficient": getattr(self, "_stoichiometric_coefficient_" + comp)(g),
                          }
            pp.initialize_data(g,
                               d,
                               getattr(self, comp + "_parameter_key"),
                               parameters,
                               )

    def _bc_type_b(self, g: pp.Grid) -> pp.BoundaryCondition:
        """Dirichlet conditions on all external boundaries."""
        # Define boundary regions
        all_bf, *_ = self._domain_boundary_sides(g)
        # Define boundary condition on faces
        return pp.BoundaryCondition(g, all_bf, "dir")

    def _bc_values_b(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous boundary values.

        Units:
            Dirichlet conditions: Pa = kg / m^1 / s^2
            Neumann conditions: m^3 / s
        """
        return np.zeros(g.num_faces)

    def _stoichiometric_coefficient_b(self, g: pp.Grid) -> int:
        """Stoichiometric coefficient for the chemical reaction

        Args:
            g: pp.Grid

        Returns:
            np.ndarray: (g.num_cells)

        """
        return self._beta * np.ones(g.num_cells)



    def _source_b(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous values except at the domain center."""

        return self._domain_center_source(g, 0.55) * g.cell_volumes

    def _initial_b(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous initial values.

        Args:
            g: pp.Grid of the subdomain.

        Returns:
            vals: np.ndarray of cell-wise concentration values
        """
        return 0.5 * np.ones(g.num_cells)


class ShearDilation(BiotNonlinearTpfa):
    def _bc_values_mechanics(self, g: pp.Grid) -> np.ndarray:
        vals = np.zeros((self._Nd, g.num_faces))
        faces = g.face_centers[self._Nd - 1] > 1 - 1e-7
        vals[self._Nd - 1, faces] = -1e-3
        vals[self._Nd - 2, faces] = 2e-3
        return vals.ravel(order="F")

    def _dilation_angle(self, g):
        return .2 * np.ones(g.num_cells)

    def _friction_coefficient(self, g):
        return 5e-1 * np.ones(g.num_cells)

    def _source_scalar(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous Dirichlet values except at left end of fracture."""
        vals = np.zeros(g.num_cells)
        if g.dim == self._Nd - 1:
            pt = np.zeros(3)
            ind = g.closest_cell(pt)
            val = 0.01
            if g.dim == 1:
                val = 8e-3
            vals[ind] = val
        return vals


class SquarePermeability2d(SquareRootPermeability):
    def _source(self, g: pp.Grid) -> np.ndarray:
        """Source term.

        Units: m^3 / s
        """
        x = g.cell_centers[0]
        y = g.cell_centers[1]
        val = (
                -2 * x * (1 - x) * (x ** 2 + 0.1)
                + 2 * x * (-x * y * (1 - y) + y * (1 - x) * (1 - y))
                - 2 * y * (1 - y) * (x ** 2 + 0.1)
        )
        return val * g.cell_volumes

    def p_analytical(self, g=None):
        if g is None:
            g = self.gb.grids_of_dimension(self.gb.dim_max())[0]
        x = g.cell_centers[0]
        y = g.cell_centers[1]
        return x * y * (1 - x) * (1 - y)

    def _permeability_function(self, pressure: pp.ad.Variable):
        val = pressure ** 2 + 0.1
        return val

    def _permeability_function_ad(self, pressure: pp.ad.Variable):
        val = pressure ** 2 + 0.1
        return val


class EquilibriumChemistry(Chemistry):
    def __init__(self, params: Dict):
        super().__init__(params)
        self.component_b_variable = "B"
        self.b_parameter_key = "chemistry_parameters_b"
        # Stoichiometric constant
        self._beta = self.params.get("beta", 1)
        self._equilibrium_constant = self.params.get("equilibrium_constant", .25)

    def _initial_condition(self):
        """Initial condition for all primary variables and the parameter representing the
         secondary flux variable (needs to be present for first call to upwind discretization).
        """
        # Pressure, a, b and flux for a-parameter
        super()._initial_condition()
        # add b and flux for b-parameter
        for g, d in self.gb:
            state = {self.component_b_variable: self._initial_b(g)}
            d[pp.STATE].update(state)
            d[pp.STATE][pp.ITERATE].update(state.copy())
            pp.initialize_data(g, d, self.b_parameter_key, {"darcy_flux": np.zeros(g.num_faces)})

    def _assign_variables(self) -> None:
        """
        Assign primary variables to the nodes and edges of the grid bucket.
        """
        super()._assign_variables()
        # First for the nodes
        for g, d in self.gb:
            d[pp.PRIMARY_VARIABLES].update({self.component_b_variable: {"cells": 1}})

    def _create_ad_variables(self) -> None:
        """Create the merged variables for potential and mortar flux"""
        super()._create_ad_variables()
        grid_list = [g for g, _ in self.gb.nodes()]
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

        subdomain_transport_equation_a: pp.ad.Operator = self._subdomain_transport_equation(self.grid_list,
                                                                                            component="a")
        subdomain_transport_equation_b: pp.ad.Operator = self._subdomain_transport_equation(self.grid_list,
                                                                                            component="b")
        equilibrium_equation: pp.ad.Operator = self._equilibrium_equation(self.grid_list)
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
        eq: pp.ad.Operator = log(equilibrium_constant) - self._alpha * log(self._ad.component_a) - self._beta * log(
            self._ad.component_b)

        return eq

    def before_newton_iteration(self):
        super().before_newton_iteration()
        self._ad.dummy_upwind_for_discretization_b.discretize(self.gb)

    def compute_fluxes(self) -> None:
        super().compute_fluxes()
        # Copy to b parameters
        for _, d in self.gb:
            d[pp.PARAMETERS][self.b_parameter_key]["darcy_flux"] = d[pp.PARAMETERS][self.a_parameter_key]["darcy_flux"]

    def _export_names(self):
        return super()._export_names() + [self.component_b_variable]