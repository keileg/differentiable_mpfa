"""
Parameters and run script for Section 4.1

Model used is SquareRootPermeability2d:
    Incompressible flow with k(p)=sqrt(p) + k_0.
Multiple runs with different grids.
"""
from typing import Callable

import numpy as np
import porepy as pp

from common_models import DataSaving, Geometry, SolutionStrategyMixin, solid_values
from constitutive_laws import DifferentiatedDarcyLaw
from grids import two_dimensional_cartesian, two_dimensional_cartesian_perturbed
from utility_functions import run_simulation_pairs_varying_parameters


def sqrt(var):
    return var ** (1 / 2)


class SquareRootPermeability:
    """Incompressible flow with :math:`k(p)=sqrt(p) + k_0`.

    The model is defined on a unit square, with Dirichlet boundary conditions on all
    boundaries. The source term is chosen such that the analytical solution is
    :math:`x * y * (1 - x) * (1 - y) + p_0`, where :math:`p_0` is the pressure at the
    boundary.

    """

    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid."""

    solid: pp.SolidConstants
    """Solid parameters."""

    fluid: pp.FluidConstants
    """Fluid parameters."""

    equation_system: pp.EquationSystem
    """Equation system manager."""

    pressure: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Pressure."""

    pressure_variable: str
    """Name of the pressure variable."""

    nd: int
    """Number of dimensions."""

    def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Pressure-dependent permeability function (not ad).

        Parameters:
            pressure: Cell-wise pressure.

        Returns:
            Cell-wise permeability values.

        """
        k0 = pp.ad.Scalar(self.solid.permeability())
        val = self.pressure(subdomains) ** (1 / 2) + k0
        return val

    def initial_condition(self) -> None:
        """Set initial guess for the variables."""
        super().initial_condition()
        vals = np.empty(0)
        for sd in self.mdg.subdomains():
            vals = np.hstack((vals, self.initial_pressure(sd)))
        self.equation_system.set_variable_values(
            vals, [self.pressure_variable], to_state=True, to_iterate=True
        )

    def initial_pressure(self, sd: pp.Grid):
        val = self.p_analytical(sd) / 2
        return val

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Source term.

        Units: m^3 / s

        Parameters:
            sd: Subdomain grid.

        Returns:
            Cell-wise source values.

        """
        sd = subdomains[0]
        x = sd.cell_centers[0]
        y = sd.cell_centers[1]
        k0 = self.solid.permeability()
        p0 = self.fluid.pressure()
        val = (
            -2 * x * (1 - x) * sqrt(k0 + x * y * (1 - x) * (1 - y) + p0)
            - 2 * y * (1 - y) * sqrt(k0 + x * y * (1 - x) * (1 - y) + p0)
            + (-x * y * (1 - x) + x * (1 - x) * (1 - y))
            * (-x * y * (1 - x) / 2 + x * (1 - x) * (1 - y) / 2)
            / sqrt(k0 + x * y * (1 - x) * (1 - y) + p0)
            + (-x * y * (1 - y) + y * (1 - x) * (1 - y))
            * (-x * y * (1 - y) / 2 + y * (1 - x) * (1 - y) / 2)
            / sqrt(k0 + x * y * (1 - x) * (1 - y) + p0)
        )

        source = pp.wrap_as_ad_array(-val * sd.cell_volumes, name="source")
        return source

    def p_analytical(self, sd=None):
        """Analytical pressure solution.

        Units: Pa.

        Parameters:
            sd: Subdomain grid.

        Returns:
            Cell-wise analytical pressure values.

        """
        if sd is None:
            sd = self.mdg.subdomains(dim=self.mdg.dim_max())[0]
        x = sd.cell_centers[0]
        y = sd.cell_centers[1]
        return x * y * (1 - x) * (1 - y) + self.fluid.pressure()

    def bc_values_darcy(self, subdomains: list[pp.Grid]) -> pp.ad.Array:
        """Boundary pressure values.

        Parameters:
            sd: Subdomain grid.

        Returns:
            Face-wise boundary pressure values.

        """
        if len(subdomains) == 0:
            return pp.wrap_as_ad_array(0, size=0)
        values = []
        for sd in subdomains:
            val = np.zeros(sd.num_faces)
            faces, *_ = self.domain_boundary_sides(sd)
            val[faces] = self.fluid.pressure()
            values.append(val)
        bc_values = pp.wrap_as_ad_array(np.concatenate(values), name="bc_values")
        return bc_values


class CombinedModel(
    SquareRootPermeability,
    Geometry,
    DataSaving,
    SolutionStrategyMixin,
    DifferentiatedDarcyLaw,
    pp.fluid_mass_balance.SinglePhaseFlow,
):
    pass


if __name__ == "__main__":
    solid_values["permeability"] = 1e-2
    params = {
        "use_ad": True,
        "use_tpfa": False,
        "grid_method": two_dimensional_cartesian,
        "n_cells": [100, 100],
        "compute_permeability_errors": False,
        "material_constants": {"solid": pp.SolidConstants(solid_values)},
    }
    update_params_small = {
        "1": {
            "legend_title": "# cells",
            "n_cells": [3, 4],
            "plotting_file_name": "verification_small",
        },
        "2": {
            "n_cells": [5, 6],
            "plotting_file_name": "verification_small",
        },
    }
    update_params_cell_number = {
        "100": {
            "legend_title": "# cells",
            "max_iterations": 40,
            "n_cells": [10, 10],
            "max_iterations": 40,
        },
        "2500": {"n_cells": [25, 25]},
        "10k": {"n_cells": [100, 100]},
        # "160k": {"n_cells": [400, 400]},  # TODO: Increase to 1m
    }
    for k in update_params_cell_number.keys():
        update_params_cell_number[k]["plotting_file_name"] = "verification_cell_number"
    update_params_cell_number_not_converging = {
        "100": {
            "legend_title": "# cells",
            "max_iterations": 24,
            "n_cells": [10, 10],
            "compute_permeability_errors": True,
        },
        "2500": {"n_cells": [25, 25]},
        "10k": {"n_cells": [100, 100]},
    }
    for k in update_params_cell_number_not_converging.keys():
        update_params_cell_number_not_converging[k][
            "plotting_file_name"
        ] = "verification_cell_number"

    update_params_mesh_type = {
        "Tetrahedra MP": {
            "simplex": True,
            "mesh_args": {
                "mesh_size_bound": 3e-2,  # 92562 at 5e-3.
                "mesh_size_frac": 1e-3,
            },
            "legend_title": "Scheme",
            "legend_label": "MP",
        },
        "Tetrahedra TP": {
            "use_tpfa": True,
            "legend_label": "TP",
        },
        "Perturbed Cartesian MP": {
            "use_tpfa": False,
            "grid_method": two_dimensional_cartesian_perturbed,
            "n_cells": [50, 50],
            "legend_label": "MP",
        },
        "Perturbed Cartesian TP": {
            "use_tpfa": True,
            "legend_label": "TP",
        },
    }
    for k in update_params_mesh_type.keys():
        update_params_mesh_type[k][
            "plotting_file_name"
        ] = "verification_mesh_and_discretization"
    update_params_anisotropy = {
        "50": {
            "use_tpfa": True,
            "legend_title": "# cells y",
            "grid_method": two_dimensional_cartesian_perturbed,
            "n_cells": [50, 50],
        },
        "200": {"n_cells": [50, 200]},
        "1000": {"n_cells": [50, 1000]},
    }
    for k in update_params_anisotropy.keys():
        update_params_anisotropy[k]["plotting_file_name"] = "verification_anisotropy"

    all_updates = [
        # update_params_small,
        update_params_cell_number,
        update_params_cell_number_not_converging,
        update_params_mesh_type,
        update_params_anisotropy,
    ]
    for up in all_updates:
        run_simulation_pairs_varying_parameters(params, up, CombinedModel)

    # plot_all_permeability_errors(update_params_cell_number_not_converging)
