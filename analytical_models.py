import porepy as pp
import numpy as np
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
        val = sqrt(pressure + 1)
        return val

    def _permeability_function_ad(self, pressure: pp.ad.Variable):
        val = sqrt(pressure + 1)
        return val

    def _initial_condition(self) -> None:
        """Set initial guess for the variables."""
        vals = np.empty((0))
        for g, _ in self.gb:
            vals = np.hstack((vals, self._initial_pressure(g)))
        self.dof_manager.distribute_variable(vals)
        self.dof_manager.distribute_variable(vals, to_iterate=True)

    def _initial_pressure(self, g):
        # return .9 * np.ones(g.num_cells)
        val = self.p_analytical(g) / (1.1 - np.random.rand(g.num_cells) / 5)
        return val

class SquareRootPermeability2d(SquareRootPermeability):
    def _source(self, g: pp.Grid) -> np.ndarray:
        """Source term.

        Units: m^3 / s
        """
        x = g.cell_centers[0]
        y = g.cell_centers[1]
        val = (
                -2*x*(1 - x)*sqrt(x*y*(1 - x)*(1 - y) + 1) - 2*y*(1 - y)*sqrt(x*y*(1 - x)*(1 - y) + 1)
                + (-x*y*(1 - x) + x*(1 - x)*(1 - y))*(-x*y*(1 - x)/2 + x*(1 - x)*(1 - y)/2)/sqrt(x*y*(1 - x)*(1 - y) + 1)
                + (-x*y*(1 - y) + y*(1 - x)*(1 - y))*(-x*y*(1 - y)/2 + y*(1 - x)*(1 - y)/2)/sqrt(x*y*(1 - x)*(1 - y) + 1)
                )
        return val * g.cell_volumes

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
class CombinedModel(SquarePermeability2d, NonlinearIncompressibleFlow):
    pass
if __name__ == "__main__":
    num_iterations = 10
    params = {
        "use_tpfa": True,
        "linear_solver": "pypardiso",
        "max_iterations": num_iterations,
        "nl_convergence_tol": 1e-10,
        "nl_divergence_tol": 1e5,
        "grid_method": two_dimensional_cartesian_perturbed,
        "n_cells": [5, 50],
    }
    nonlinear_model = CombinedModel(params)
    pp.run_stationary_model(nonlinear_model, params)
    params_linear = params.copy()
    params_linear["use_linear_discretization"] = True
    linear_model = CombinedModel(params_linear)
    pp.run_stationary_model(linear_model, params_linear)
    plot_convergence(nonlinear_model, linear_model, plot_errors=True)
