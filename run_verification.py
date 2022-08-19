"""
Verification runs.
Model used is SquareRootPermeability2d:
    Incompressible flow with k(p)=sqrt(p) + k_0.
Multiple runs with different grids. Mostly mpfa, two runs with tpfa.
"""
import numpy as np
import scipy.sparse as sps
from grids import two_dimensional_cartesian, two_dimensional_cartesian_perturbed
from matplotlib import pyplot as plt
from models import NonlinearIncompressibleFlow
from utility_functions import (
    plot_all_permeability_errors,
    run_simulation_pairs_varying_parameters,
)

import porepy as pp


def sqrt(var):
    return var ** (1 / 2)


class SquareRootPermeability:
    def _source(self, sd: pp.Grid) -> np.ndarray:
        """Source term.

        Units: m^3 / s
        """
        x = sd.cell_centers[0]
        val = (1 / 2 - x) * (1 - 2 * x) / sqrt(x * (1 - x) + 1) - 2 * sqrt(
            x * (1 - x) + 1
        )
        return val * sd.cell_volumes

    def p_analytical(self, sd=None):
        if sd is None:
            sd = self.mdg.subdomains(dim=self.mdg.dim_max())[0]
        x = sd.cell_centers[0]
        return x * (1 - x)

    def _permeability_function(self, pressure: pp.ad.Variable):
        k0 = self.params["k0"]
        val = np.sqrt(pressure) + k0
        return val

    def _permeability_function_ad(self, pressure: pp.ad.Variable):
        nc = pressure.val.size
        k0 = pp.ad.Ad_array(
            self.params["k0"] * np.ones(nc),
            sps.csr_matrix((nc, self.dof_manager.num_dofs()), dtype=float),
        )
        val = np.sqrt(pressure.val) + self.params["k0"]
        diff = 0.5 / np.sqrt(pressure.val)
        jac = pressure.jac.copy()
        jac.data = diff
        K = sqrt(pressure) + k0
        return pp.ad.Ad_array(val, jac)

    def _initial_condition(self) -> None:
        """Set initial guess for the variables."""
        vals = np.empty(0)
        for sd in self.mdg.subdomains():
            vals = np.hstack((vals, self._initial_pressure(sd)))
        self.dof_manager.distribute_variable(vals)
        self.dof_manager.distribute_variable(vals, to_iterate=True)

    def _initial_pressure(self, sd: pp.Grid):
        # return self.params["p0"] * np.ones(g.num_cells)
        val = self.p_analytical(sd) / (1.1 - np.random.rand(sd.num_cells) / 5)
        # val = self.p_analytical(sd) / (1.1)  # - np.random.rand(g.num_cells) / 5)
        return val


class SquareRootPermeability2d(SquareRootPermeability):
    def _source(self, sd: pp.Grid) -> np.ndarray:
        """Source term.

        Units: m^3 / s
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
        if sd is None:
            sd = self.mdg.subdomains(dim=self.mdg.dim_max())[0]
        x = sd.cell_centers[0]
        y = sd.cell_centers[1]
        return x * y * (1 - x) * (1 - y) + self.params["p0"]

    def _bc_values(self, sd: pp.Grid):
        val = np.zeros(sd.num_faces)
        faces, *_ = self._domain_boundary_sides(sd)
        val[faces] = self.params["p0"]
        return val


class LinearPermeability2d:
    """Simple linear permeability.
    K(p) = k0 + p

    Not used in the paper at the moment, should probably be moved to archive.
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


class CombinedModel(SquareRootPermeability2d, NonlinearIncompressibleFlow):
    pass


class Linear(LinearPermeability2d, NonlinearIncompressibleFlow):
    pass


if __name__ == "__main__":
    params = {
        "use_ad": True,
        "use_tpfa": False,
        "grid_method": two_dimensional_cartesian,
        "n_cells": [100, 100],
        "k0": 1e-2,
        "p0": 0.0,
        # "max_iterations": 40,
        "plotting_file_name": "verification_cell_number",
    }
    update_params_small = {
        "1": {"legend_title": "# cells", "n_cells": [3, 4]},
    }
    update_params_cell_number = {
        "100": {"legend_title": "# cells", "n_cells": [10, 10]},
        "2500": {"n_cells": [25, 25]},
        "10k": {"n_cells": [100, 100]},
        # "160k": {"n_cells": [400, 400]},  # TODO: Increase to 1m
    }
    update_params_mesh_type = {
        "Cart, MP": {
            "legend_title": "Mesh, discretization",
            "plotting_file_name": "verification_mesh_and_discretization",
            "n_cells": [10, 10],
            # "n_cells": [100, 100],
        },
        "Tri, MP": {
            "simplex": True,
            "mesh_args": {
                "mesh_size_bound": 5e-2,  # 92562 at 5e-3.
                "mesh_size_frac": 1e-3,
            },
        },
        "Tri, TP": {"use_tpfa": True},
        "Pert, MP": {
            "use_tpfa": False,
            "grid_method": two_dimensional_cartesian_perturbed,
        },
        "Pert, TP": {"use_tpfa": True},
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
        update_params_cell_number,
        # update_params_mesh_type,
        # update_params_anisotropy,
    ]
    for up in all_updates:
        run_simulation_pairs_varying_parameters(params, up, CombinedModel)

    plot_all_permeability_errors(update_params_cell_number)
