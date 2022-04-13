import porepy as pp
import numpy as np
import scipy.sparse as sps
from typing import Dict
from grids import one_dimensional_grid_bucket, two_dimensional_cartesian, two_dimensional_cartesian_perturbed
from incompressible_flow_nonlinear_tpfa import NonlinearIncompressibleFlow
from plotting import plot_convergence


def sqrt(var):
    return var ** (1 / 2)

class SquareRootPermeability:
    def _source(self, g: pp.Grid) -> np.ndarray:
        """Source term.

        Units: m^3 / s
        """
        x = g.cell_centers[0]
        val = (1/2 - x)*(1 - 2*x)/sqrt(x*(1 - x) + 1) - 2*sqrt(x*(1 - x) + 1)
        return val * g.cell_volumes

    def p_analytical(self, g=None):
        if g is None:
            g = self.gb.grids_of_dimension(self.gb.dim_max())[0]
        x = g.cell_centers[0]
        return x * (1 - x)

    def _permeability_function(self, pressure: pp.ad.Variable):
        k0 = self.params["k0"]
        val = sqrt(pressure + k0)
        return val

    def _permeability_function_ad(self, pressure: pp.ad.Variable):
        nc = pressure.val.size
        k0 = pp.ad.Ad_array(self.params["k0"] * np.ones(nc), sps.csr_matrix((nc, self.dof_manager.num_dofs()), dtype=float))
        val = sqrt(pressure + k0)
        return val

    def _initial_condition(self) -> None:
        """Set initial guess for the variables."""
        vals = np.empty((0))
        for g, _ in self.gb:
            vals = np.hstack((vals, self._initial_pressure(g)))
        self.dof_manager.distribute_variable(vals)
        self.dof_manager.distribute_variable(vals, to_iterate=True)

    def _initial_pressure(self, g):
        # return np.ones(g.num_cells)
        val = self.p_analytical(g) / (1.1 - np.random.rand(g.num_cells) / 5)
        return val

class SquareRootPermeability2d(SquareRootPermeability):
    def _source(self, g: pp.Grid) -> np.ndarray:
        """Source term.

        Units: m^3 / s
        """
        x = g.cell_centers[0]
        y = g.cell_centers[1]
        k0 = self.params["k0"]
        val = -2*x*(1 - x)*sqrt(k0 + x*y*(1 - x)*(1 - y)) - 2*y*(1 - y)*sqrt(k0 + x*y*(1 - x)*(1 - y)) + (-x*y*(1 - x) + x*(1 - x)*(1 - y))*(-x*y*(1 - x)/2 + x*(1 - x)*(1 - y)/2)/sqrt(k0 + x*y*(1 - x)*(1 - y)) + (-x*y*(1 - y) + y*(1 - x)*(1 - y))*(-x*y*(1 - y)/2 + y*(1 - x)*(1 - y)/2)/sqrt(k0 + x*y*(1 - x)*(1 - y))

        return -val * g.cell_volumes

    def p_analytical(self, g=None):
        if g is None:
            g = self.gb.grids_of_dimension(self.gb.dim_max())[0]
        x = g.cell_centers[0]
        y = g.cell_centers[1]
        return x * y * (1 - x) * (1 - y)

class SquarePermeability2d(SquareRootPermeability):
    def _source(self, g: pp.Grid) -> np.ndarray:
        """Source term.

        Units: m^3 / s
        """
        x = g.cell_centers[0]
        y = g.cell_centers[1]
        val = -2*x*(1 - x)*(x**2 + 0.1) + 2*x*(-x*y*(1 - y) + y*(1 - x)*(1 - y)) - 2*y*(1 - y)*(x**2 + 0.1)
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



class CombinedModel(SquareRootPermeability2d, NonlinearIncompressibleFlow):
    pass
# class CombinedModel(SquarePermeability2d, NonlinearIncompressibleFlow):
#     pass

def run_linear_and_nonlinear(params: Dict, model_class) -> None:
    """Run a pair of one linear and one nonlinear simulation.

    Args:
        params: Setup and run parameters
        model_class: Model to be used for the simulations

    Returns:
        nonlinear_model, linear_model

    """
    params_linear = params.copy()
    params_linear["use_linear_discretization"] = True
    linear_model = model_class(params_linear)
    pp.run_stationary_model(linear_model, params_linear)
    params["file_name"] += "_nonlinear"
    nonlinear_model = model_class(params)
    pp.run_stationary_model(nonlinear_model, params)

    plot_convergence(nonlinear_model, linear_model, plot_errors=False)
    return nonlinear_model, linear_model

def run_simulation_pairs_varying_parameters(params: Dict, update_params: Dict[str, Dict], model_class) -> None:
    """Run multiple pairs of simulations varying some parameters.

    Args:
        params: Dictionary containing initial setup and run parameters
        update_params: Dictionary with keys identifying each simulation pair and
            values being dictionaries specifying the updates to the initial parameters.

    Example usage:
    The following parameters produce four simulations (linear + 100 cells,
    nonlinear + 100 cells, linear + 4 cells and nonlinear + 4 cells):

        params = {"foo": "bar", "grid_method": two_dimensional_cartesian, ...}
        update_params = {"hundred_cells": {"n_cells": [10, 10]},
                         "four_cells": {"n_cells": [2, 2]},
                         }


    """
    for name, updates in update_params.items():
        params.update(updates)
        params["file_name"] = name
        run_linear_and_nonlinear(params, model_class)
if __name__ == "__main__":
    num_iterations = 10
    params = {
        "use_tpfa": False,
        "linear_solver": "pypardiso",
        "max_iterations": num_iterations,
        "nl_convergence_tol": 1e-10,
        "nl_divergence_tol": 1e5,
        "grid_method": two_dimensional_cartesian,
        "n_cells": [100, 100],
        "k0": 1e-3,
        "plotting_file_name": "analytical_solution_mpfa_varying_cell_number"
    }
    update_params = {
        "100 cells": {"n_cells": [10, 10]},
        "10000 cells": {"n_cells": [100, 100]},
        "250000 cells": {"n_cells": [500, 500]},
    }
    run_simulation_pairs_varying_parameters(params, update_params, CombinedModel)
