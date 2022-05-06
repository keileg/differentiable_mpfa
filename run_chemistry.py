import logging

import numpy as np
import porepy as pp
from models import Chemistry

from grids import three_dimensional_cartesian
from utility_functions import run_simulation_pairs_varying_parameters, plot_multiple_time_steps

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ChemistryParameters:
    def _set_chemistry_parameters(self) -> None:
        """Set parameters for chemical model.

        Returns:

        """
        for g, d in self.gb:
            for comp in ["a", "c"]:
                parameters = {"mass_weight": np.ones(g.num_cells)}
                if comp == "c":
                    parameters.update({"density_inv": 1 / self._density_c(g),
                                       "reference_porosity": self.rock.POROSITY,
                                       "reference_reaction_rate": self._reference_reaction_rate(g),
                                       "equilibrium_constant_inv": np.ones(g.num_cells) / self._equilibrium_constant,
                                       }
                                      )

                else:
                    # Else covers only "a" but can be used for multiple dissolved components.
                    parameters.update({
                        "bc": getattr(self, "_bc_type_" + comp)(g),
                        "bc_values": getattr(self, "_bc_values_" + comp)(g),
                        "source": getattr(self, "_source_" + comp)(g),
                        "stoichiometric_coefficient": getattr(self, "_stoichiometric_coefficient_" + comp)(g),
                    }
                    )
                pp.initialize_data(g,
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

    def _density_c(self, g):
        """Density of the c component.

        Args:
            g: pp.Grid

        Returns:
            np.ndarray: (g.num_cells)

        """
        return self.params.get("density", 4.e1) * np.ones(g.num_cells)

    def _source_a(self, g: pp.Grid) -> np.ndarray:
        """Component a source.

        Args:
            g: Grid representing a subdomain.

        Returns:
            source_vals: array of cell-wise values.
        Homogeneous values except at the domain center.
        """
        source_vals = self._domain_center_source(g, val=.3 * 1e2) * g.cell_volumes
        return source_vals

    def _source(self, g: pp.Grid) -> np.ndarray:
        """Flow source.

        Args:
            g: Grid representing a subdomain.

        Returns:
            source_vals: array of cell-wise values.
        Homogeneous values except at the domain center.
        """
        source_vals = self._domain_center_source(g, val=1e2) * g.cell_volumes
        return source_vals

    def _initial_a(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous initial values.

        Args:
            g: pp.Grid of the subdomain.

        Returns:
            vals: np.ndarray of cell-wise concentration values
        """
        return 0.5 * np.ones(g.num_cells)

    def _initial_c(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous initial values.

        Args:
            g: pp.Grid of the subdomain.

        Returns:
            vals: np.ndarray of cell-wise concentration values
        """
        return 0.8 * np.ones(g.num_cells)

    def _initial_pressure(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous initial values.

        Args:
            g: pp.Grid of the subdomain.

        Returns:
            vals: np.ndarray of cell-wise pressure values
        """
        return 1e4 * np.ones(g.num_cells)

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
        if self.time_index < self.params["n_time_steps"] - 1:
            self.time_step = 5e-1
            # - self.time_step#self.end_time / 10 ** (self.params["n_time_steps"]-self.time_index) #*= 1e1
        else:
            self.time_step = self.end_time - self.time + 1e-10
        print(self.time_step)
        self._ad.time_step._value = self.time_step


class CombinedModel(ChemistryParameters, Chemistry):
    pass


if __name__ == "__main__":
    nc = 15
    params = {
        "use_tpfa": True,
        "grid_method": three_dimensional_cartesian,
        "plotting_file_name": "chemistry",
        "file_name": "chemistry",
        "time_step": 1e-3,
        "end_time": 1.501e0,
        "n_cells": [nc, nc, nc],
        "n_time_steps": 3,
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
    run_simulation_pairs_varying_parameters(
        params, update_params, CombinedModel
    )

    plot_multiple_time_steps(updates=update_params, n_steps=params["n_time_steps"])
    for k in update_params.keys():
        update_params[k]["models"][0].params["plotting_file_name"] += "_linear"
    plot_multiple_time_steps(updates=update_params, n_steps=params["n_time_steps"], model_type="linear")
