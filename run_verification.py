"""
Parameters and run script for Section 4.1

Model used is SquareRootPermeability2d:
    Incompressible flow with k(p)=sqrt(p) + k_0.
Multiple runs with different grids.
"""
from typing import Optional

import numpy as np
import porepy as pp
import scipy.sparse as sps

from grids import two_dimensional_cartesian, two_dimensional_cartesian_perturbed
from models import NonlinearIncompressibleFlow
from utility_functions import (
    plot_all_permeability_errors,
    run_simulation_pairs_varying_parameters,
)


def sqrt(var):
    return var ** (1 / 2)


class SquareRootPermeability:
    def _source(self, sd: pp.Grid) -> np.ndarray:
        """Source term.

        Units: m^3 / s

        Args:
            sd: Subdomain grid.

        Returns:
            Cell-wise source values.
        """
        x = sd.cell_centers[0]
        val = (1 / 2 - x) * (1 - 2 * x) / sqrt(x * (1 - x) + 1) - 2 * sqrt(
            x * (1 - x) + 1
        )
        return val * sd.cell_volumes

    def p_analytical(self, sd: Optional[pp.Grid]=None) -> np.ndarray:
        """Analytical pressure solution.

        Units: m^3 / s

        Args:
            sd: Subdomain grid.

        Returns:
            Cell-wise source values.
        """
        if sd is None:
            sd = self.mdg.subdomains(dim=self.mdg.dim_max())[0]
        x = sd.cell_centers[0]
        return x * (1 - x)

    def _permeability_function(self, pressure: np.ndarray) -> np.ndarray:
        """Pressure-dependent permeability function (not ad).

        Args:
            pressure: Cell-wise pressure.

        Returns:
            Cell-wise permeability values.

        """
        k0 = self.params["k0"]
        val = np.sqrt(pressure) + k0
        return val

    def _permeability_function_ad(self, pressure: pp.ad.Variable):
        """Pressure-dependent permeability function.

        Intended usage: Wrapping in pp.ad.Function.

        Args:
            pressure: Ad variable.

        Returns:
            Ad_array

        """
        val = np.sqrt(pressure.val) + self.params["k0"]
        diff = 0.5 / np.sqrt(pressure.val)
        jac = pressure.jac.copy()
        jac.data = diff
        return pp.ad.Ad_array(val, jac)

    def _initial_condition(self) -> None:
        """Set initial guess for the variables."""
        vals = np.empty(0)
        for sd in self.mdg.subdomains():
            vals = np.hstack((vals, self._initial_pressure(sd)))
        self.dof_manager.distribute_variable(vals)
        self.dof_manager.distribute_variable(vals, to_iterate=True)

    def _initial_pressure(self, sd: pp.Grid):
        val = self.p_analytical(sd) / 2 #(10.1 - np.random.rand(sd.num_cells) / 5)
        # val = np.ones(sd.num_cells) / 1
        return val


class SquareRootPermeability2d(SquareRootPermeability):
    def _source(self, sd: pp.Grid) -> np.ndarray:
        """Source term.

        Units: m^3 / s

        Args:
            sd: Subdomain grid.

        Returns:
            Cell-wise source values.
        """
        x = sd.cell_centers[0]
        y = sd.cell_centers[1]
        k0 = self.params["k0"]
        p0 = self.params["p0"]
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

        return -val * sd.cell_volumes

    def p_analytical(self, sd=None):
        """Analytical pressure solution.

        Units: m^3 / s

        Args:
            sd: Subdomain grid.

        Returns:
            Cell-wise source values.
        """
        if sd is None:
            sd = self.mdg.subdomains(dim=self.mdg.dim_max())[0]
        x = sd.cell_centers[0]
        y = sd.cell_centers[1]
        return x * y * (1 - x) * (1 - y) + self.params["p0"]

    def _bc_values(self, sd: pp.Grid) -> np.ndarray:
        """Boundary pressure values.

        Args:
            sd: Subdomain grid.

        Returns:
            Face-wise boundary pressure values.
        """
        val = np.zeros(sd.num_faces)
        faces, *_ = self._domain_boundary_sides(sd)
        val[faces] = self.params["p0"]
        return val


class CombinedModel(SquareRootPermeability2d, NonlinearIncompressibleFlow):
    pass


if __name__ == "__main__":
    params = {
        "use_ad": True,
        "use_tpfa": False,
        "grid_method": two_dimensional_cartesian,
        "n_cells": [100, 100],
        "k0": 1e-2,
        "p0": 0.0,
        "compute_permeability_errors": False,
    }
    update_params_small = {
        "1": {"legend_title": "# cells", "n_cells": [3, 4]},
    }
    update_params_cell_number = {
        "100": {
            "legend_title": "# cells",
            "max_iterations": 40,
            "n_cells": [10, 10],
            "plotting_file_name": "verification_cell_number_converging",
            "max_iterations": 40,
        },
        "2500": {"n_cells": [25, 25]},
        "10k": {"n_cells": [100, 100]},
        # "160k": {"n_cells": [400, 400]},  # TODO: Increase to 1m
    }
    update_params_cell_number_not_converging = {
        "100": {
            "legend_title": "# cells",
            "max_iterations": 24,
            "n_cells": [10, 10],
            "plotting_file_name": "verification_cell_number",
            "compute_permeability_errors": True,
        },
        "2500": {"n_cells": [25, 25]},
        "10k": {"n_cells": [100, 100]},
    }

    update_params_mesh_type = {
        "Tetrahedra MP": {
            "simplex": True,
            "mesh_args": {
                "mesh_size_bound": 3e-2,  # 92562 at 5e-3.
                "mesh_size_frac": 1e-3,
            },
            "legend_title": "Scheme",
            "legend_label": "MP",
            "plotting_file_name": "verification_mesh_and_discretization",
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
    update_params_anisotropy = {
        "50": {
            "use_tpfa": True,
            "plotting_file_name": "verification_anisotropy",
            "legend_title": "# cells y",
            "grid_method": two_dimensional_cartesian_perturbed,
            "n_cells": [50, 50],
        },
        "200": {"n_cells": [50, 200]},
        "1000": {"n_cells": [50, 1000]},
    }

    all_updates = [
        # update_params_cell_number,
        # update_params_cell_number_not_converging,
        update_params_mesh_type,
        # update_params_anisotropy,
    ]
    for up in all_updates:
        run_simulation_pairs_varying_parameters(params, up, CombinedModel)

    plot_all_permeability_errors(update_params_cell_number_not_converging)
