import logging

import numpy as np
from grids import three_dimensional_cartesian
from models import Chemistry
from utility_functions import (plot_multiple_time_steps,
                               run_simulation_pairs_varying_parameters)

import porepy as pp

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ChemistryParameters:
    def _set_chemistry_parameters(self) -> None:
        """Set parameters for chemical model.

        Returns:

        """
        for g, d in self.gb:
            for comp in ["a", "c"]:
                parameters = {
                    "mass_weight": np.ones(g.num_cells),
                    "density_inv": 1 / getattr(self, "_density_" + comp)(g),
                }
                if comp == "c":
                    parameters.update(
                        {
                            "reference_porosity": self.rock.POROSITY
                            * np.ones(g.num_cells),
                            "reference_reaction_rate": self._reference_reaction_rate(g),
                            "equilibrium_constant_inv": np.ones(g.num_cells)
                            / self._equilibrium_constant,
                        }
                    )

                else:
                    # Else covers only "a" but can be used for multiple dissolved components.
                    parameters.update(
                        {
                            "bc": getattr(self, "_bc_type_" + comp)(g),
                            "bc_values": getattr(self, "_bc_values_" + comp)(g),
                            "source": getattr(self, "_source_" + comp)(g),
                            "stoichiometric_coefficient": getattr(
                                self, "_stoichiometric_coefficient_" + comp
                            )(g),
                        }
                    )
                pp.initialize_data(
                    g,
                    d,
                    getattr(self, comp + "_parameter_key"),
                    parameters,
                )

    def _bc_type_a(self, g: pp.Grid) -> pp.BoundaryCondition:
        """Dirichlet conditions on all external boundaries.

        Returns:
            pp.BoundaryCondition:
        """
        # Define boundary regions
        all_bf, *_ = self._domain_boundary_sides(g)
        # Define boundary condition on faces
        return pp.BoundaryCondition(g, all_bf, "dir")

    def _bc_values_a(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous boundary values.

        Units:
            Dirichlet conditions: Pa = kg / m^1 / s^2
            Neumann conditions: m^3 / s
        """
        return np.zeros(g.num_faces)

    def _stoichiometric_coefficient_a(self, g: pp.Grid) -> int:
        """Stoichiometric coefficient for the chemical reaction

        Args:
            g: pp.Grid

        Returns:
            np.ndarray: (g.num_cells)

        """
        return self._alpha * np.ones(g.num_cells)

    def _density_a(self, g):
        """Density of the c component.

        Args:
            g: pp.Grid

        Returns:
            np.ndarray: (g.num_cells)

        """
        return np.ones(g.num_cells)

    def _density_c(self, g):
        """Density of the c component.

        Args:
            g: pp.Grid

        Returns:
            np.ndarray: (g.num_cells)

        """
        return self.params.get("density", 1) * np.ones(g.num_cells)

    def _source_a(self, g: pp.Grid) -> np.ndarray:
        """Component a source.

        Args:
            g: Grid representing a subdomain.

        Returns:
            source_vals: array of cell-wise values.
        Homogeneous values except at the domain center.
        """
        rate = self.params.get("source_rate")
        source_vals = (
            self._domain_center_source(g, val=0.2 * rate)
            * g.cell_volumes
            * self._density_a(g)
        )
        return source_vals

    def _source(self, g: pp.Grid) -> np.ndarray:
        """Flow source.

        Args:
            g: Grid representing a subdomain.

        Returns:
            source_vals: array of cell-wise values.
        Homogeneous values except at the domain center.
        """
        rate = self.params.get("source_rate")
        source_vals = self._domain_center_source(g, val=rate) * g.cell_volumes
        return source_vals

    def _initial_a(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous initial values.

        Args:
            g: pp.Grid of the subdomain.

        Returns:
            vals: np.ndarray of cell-wise concentration values
        """
        return 0.5 * self._density_a(g)

    def _initial_c(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous initial values.

        Args:
            g: pp.Grid of the subdomain.

        Returns:
            vals: np.ndarray of cell-wise concentration values
        """
        return 0.9 * self._density_c(g)

    def _initial_pressure(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous initial values.

        Args:
            g: pp.Grid of the subdomain.

        Returns:
            vals: np.ndarray of cell-wise pressure values
        """
        return 0 * np.ones(g.num_cells)

    def _reference_reaction_rate(self, g: pp.Grid) -> np.ndarray:
        """Reference rate r_0 of the reaction term.

        Args:
            g: pp.Grid representing a subdomain.

        Returns:
            r_0: array of cell-wise rates. Homogeneous values
                are definitely the obvious choice.
        """
        r_0 = self.params.get("reaction_rate", 1e-2) * np.ones(g.num_cells)
        return r_0

    def after_newton_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        super().after_newton_convergence(solution, errors, iteration_counter)
        self._adjust_time_step()

    def after_newton_failure(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        super().after_newton_failure(solution, errors, iteration_counter)
        self._adjust_time_step()

    def _adjust_time_step(self):
        if self.time_index < self.params["n_time_steps"] - 1:
            # self.time_step = 5e-1
            # self.time_step = 1e-2
            self.time_step *= 2  # self.end_time / 10 ** (self.params["n_time_steps"]-self.time_index) #*= 1e1
        else:
            self.time_step = self.end_time - self.time + 1e-10
        print(self.time_step)
        self._ad.time_step._value = self.time_step


class CombinedModel(ChemistryParameters, Chemistry):
    pass


if __name__ == "__main__":
    nc = 7
    params = {
        "use_tpfa": True,
        "grid_method": three_dimensional_cartesian,
        "plotting_file_name": "chemistry",
        "file_name": "chemistry",
        "time_step": 1e-3,
        "end_time": 1.501e0,
        "n_cells": [nc, nc, nc],
        "n_time_steps": 2,
        "permeability_exponent": 3,
    }
    update_params = {
        "1": {
            "plotting_file_name": "chemistry_temp",
            "legend_title": "reaction rate",
            "reaction_rate": 5e-2,
        },
        "2": {"reaction_rate": 5e-3},
        "3": {"reaction_rate": 5e-4},
    }
    update_params = {
        "20": {
            "plotting_file_name": "chemistry_density",
            "legend_title": "density",
            "reaction_rate": 5e-2,
            "density": 2.1e1,
        },
        "30": {"density": 3e1},
        "50": {"density": 5e1},
        "100": {"density": 10e1},
    }
    k = 2e-6
    k1 = 5
    update_params = {
        "20": {
            "plotting_file_name": "chemistry_dt",
            "legend_title": "time step",
            "reaction_rate": 1e2,
            "source_rate": 5.0e3,
            "time_step": 1 * k,
            "end_time": k1 * 1 * k,
            "n_time_steps": 4,
        },
        # "30": {"density": 3e1},
        # "50": {"source_rate": 3e2},
        "3": {
            "time_step": 95 * k,
            "end_time": k1 * 95 * k,
        },
        "10": {
            "time_step": 75 * k,
            "end_time": k1 * 75 * k,
        },
        "50": {
            "time_step": 40 * k,
            "end_time": k1 * 40 * k,
        },
    }

    rate0 = 1e4
    k = 1e-4
    update_params = {
        "1": {
            "plotting_file_name": "chemistry_source",
            "legend_title": "source_rate",
            "reaction_rate": 1e2,
            "source_rate": 1 * rate0,
            "time_step": 1 * k,
            "end_time": k1 * 1 * k,
            "n_time_steps": 2,
        },
        "2": {"source_rate": 2 * rate0},
        "4": {"source_rate": 4 * rate0},
        "8": {"source_rate": 8 * rate0},
        # "3": {"time_step": 5 * k,
        #       "end_time": k1 * 5 * k,
        #       },
        # "10": {"time_step": 15 * k,
        #        "end_time": k1 * 15 * k,
        #        },
        # "50": {"time_step": 40 * k,
        #        "end_time": k1 * 40 * k,
        #        },
    }
    rate0 = 1e0
    dt = 5e-3
    k1 = 1
    update_params = {
        "3": {
            "plotting_file_name": "chemistry_exponent",
            "legend_title": r"$\eta$",
            "reaction_rate": 2e1,
            "source_rate": rate0,
            "time_step": 1 * dt,
            "end_time": k1 * 1 * dt,
            "n_time_steps": 1,
            "permeability_exponent": 3,
        },
        "6": {"permeability_exponent": 6},
        "9": {"permeability_exponent": 9},
        "12": {"permeability_exponent": 12},
        # "15": {"permeability_exponent": 15},
    }
    rate0 = 1e0
    dt = 5e-3
    k1 = 1
    update_params = {
        "3": {
            "plotting_file_name": "chemistry_exponent",
            "legend_title": r"$\eta$",
            "reaction_rate": 2e1,
            "source_rate": rate0,
            "time_step": 1 * dt,
            "end_time": k1 * 1 * dt,
            "n_time_steps": 1,
            "permeability_exponent": 4,
        },
        # "6": {"permeability_exponent": 8},
        # "9": {"permeability_exponent": 12},
        # "12": {"permeability_exponent": 16},
        # "15": {"permeability_exponent": 15},
    }
    run_simulation_pairs_varying_parameters(params, update_params, CombinedModel)

    # plot_multiple_time_steps(updates=update_params, n_steps=params["n_time_steps"])
    # for k in update_params.keys():
    #     update_params[k]["models"][0].params["plotting_file_name"] += "_linear"
    # plot_multiple_time_steps(updates=update_params, n_steps=params["n_time_steps"], model_type="linear")
