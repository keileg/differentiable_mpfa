"""
Verification runs.
Model used is SquareRootPermeability2d:
    Incompressible flow with k(p)=sqrt(p) + k_0.
Multiple runs with different grids. Mostly mpfa, two runs with tpfa.
"""
import numpy as np
import scipy.sparse as sps
from grids import (two_dimensional_cartesian,
                   two_dimensional_cartesian_perturbed)
from models import NonlinearIncompressibleFlow
from utility_functions import run_simulation_pairs_varying_parameters

import porepy as pp


def sqrt(var):
    return var ** (1 / 2)


class SquareRootPermeability:
    def _source(self, g: pp.Grid) -> np.ndarray:
        """Source term.

        Units: m^3 / s
        """
        x = g.cell_centers[0]
        val = (1 / 2 - x) * (1 - 2 * x) / sqrt(x * (1 - x) + 1) - 2 * sqrt(
            x * (1 - x) + 1
        )
        return val * g.cell_volumes

    def p_analytical(self, g=None):
        if g is None:
            g = self.gb.grids_of_dimension(self.gb.dim_max())[0]
        x = g.cell_centers[0]
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
        vals = np.empty((0))
        for g, _ in self.gb:
            vals = np.hstack((vals, self._initial_pressure(g)))
        self.dof_manager.distribute_variable(vals)
        self.dof_manager.distribute_variable(vals, to_iterate=True)

    def _initial_pressure(self, g):
        # return self.params["p0"] * np.ones(g.num_cells)
        val = self.p_analytical(g) / (1.1 - np.random.rand(g.num_cells) / 5)
        val = self.p_analytical(g) / (1.1)  # - np.random.rand(g.num_cells) / 5)
        return val


class SquareRootPermeability2d(SquareRootPermeability):
    def _source(self, g: pp.Grid) -> np.ndarray:
        """Source term.

        Units: m^3 / s
        """
        x = g.cell_centers[0]
        y = g.cell_centers[1]
        k0 = self.params["k0"]
        p0 = self.params["p0"]
        k1 = 1
        val = (
            -2 * k1 * x * (1 - x) * sqrt(k0 + k1 * x * y * (1 - x) * (1 - y) + p0)
            - 2 * k1 * y * (1 - y) * sqrt(k0 + k1 * x * y * (1 - x) * (1 - y) + p0)
            + (-k1 * x * y * (1 - x) + k1 * x * (1 - x) * (1 - y))
            * (-k1 * x * y * (1 - x) / 2 + k1 * x * (1 - x) * (1 - y) / 2)
            / sqrt(k0 + k1 * x * y * (1 - x) * (1 - y) + p0)
            + (-k1 * x * y * (1 - y) + k1 * y * (1 - x) * (1 - y))
            * (-k1 * x * y * (1 - y) / 2 + k1 * y * (1 - x) * (1 - y) / 2)
            / sqrt(k0 + k1 * x * y * (1 - x) * (1 - y) + p0)
        )

        return -val * g.cell_volumes

    def p_analytical(self, g=None):
        if g is None:
            g = self.gb.grids_of_dimension(self.gb.dim_max())[0]
        x = g.cell_centers[0]
        y = g.cell_centers[1]
        k1 = 1
        return k1 * x * y * (1 - x) * (1 - y) + self.params["p0"]

    def _bc_values(self, g):
        val = np.zeros(g.num_faces)
        faces, *_ = self._domain_boundary_sides(g)
        val[faces] = self.params["p0"]
        return val

    def _vector_source(self, g: pp.Grid) -> np.ndarray:
        """Zero vector source (gravity).

        To assign a gravity-like vector source, add a non-zero contribution in
        the last dimension:
            vals[-1] = - pp.GRAVITY_ACCELERATION * fluid_density
        """
        vals = np.zeros((self.gb.dim_max(), g.num_cells))
        vals[-1] = -1e-1
        return vals


class LinearPermeability2d:
    """Simple linear permeability.
    K(p) = k0 + p

    Not used in the paper at the moment, should probably be moved to archive.
    """

    def _source(self, g):
        x = g.cell_centers[0]
        y = g.cell_centers[1]
        k0 = self.params["k0"]
        val = (
            -2 * x * (1 - x) * (x * y * (1 - x) * (1 - y) + k0)
            - 2 * y * (1 - y) * (x * y * (1 - x) * (1 - y) + k0)
            + (-x * y * (1 - x) + x * (1 - x) * (1 - y)) ** 2
            + (-x * y * (1 - y) + y * (1 - x) * (1 - y)) ** 2
        )
        return -val * g.cell_volumes

    def p_analytical(self, g=None):
        if g is None:
            g = self.gb.grids_of_dimension(self.gb.dim_max())[0]
        x = g.cell_centers[0]
        y = g.cell_centers[1]
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
    num_iterations = 10
    params = {
        "use_ad": True,
        "use_tpfa": False,
        "linear_solver": "pypardiso",
        "max_iterations": num_iterations,
        "nl_convergence_tol": 1e-10,
        "nl_divergence_tol": 1e5,
        "grid_method": two_dimensional_cartesian,
        "n_cells": [100, 100],
        "k0": 1e-1,
        "p0": 0,
        "plotting_file_name": "verification_cell_number",
    }

    update_params_cell_number = {
        "100": {"legend_title": "# cells", "n_cells": [10, 10]},
        "10k": {"n_cells": [100, 100]},
        "160k": {"n_cells": [400, 400]},  # TODO: Increase to 1m
    }
    update_params_mesh_type = {
        "Cart, MP": {
            "legend_title": "Mesh, discretization",
            "plotting_file_name": "verification_mesh_and_discretization",
            # "n_cells": [10, 10],
            "n_cells": [100, 100],
        },
        "Tri, MP": {
            "simplex": True,
            "mesh_args": {
                "mesh_size_bound": 5e-3,  # 92562 at 5e-3.
                "mesh_size_frac": 1e-5,
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
            "n_cells": [4, 3],
        },
        # "200": {"n_cells": [50, 200]},
        # "1000": {"n_cells": [50, 1000]},
    }
    #
    # all_updates = [update_params_cell_number, update_params_mesh_type, update_params_anisotropy]
    # for up in all_updates:
    #     run_simulation_pairs_varying_parameters(params, up, CombinedModel)

    # update_params_linear = {
    #     "4": {
    #         "plotting_file_name": "linear_permeability_cells",
    #         "legend_title": "# cells",
    #         "n_cells": [2, 2]
    #     },
    #     "100": {
    #         "n_cells": [10, 10]},
    #     "1600":
    #         {"n_cells": [40, 40]},
    # }
    run_simulation_pairs_varying_parameters(
        params, update_params_anisotropy, CombinedModel
    )
