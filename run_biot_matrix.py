import numpy as np
from grids import three_dimensional_cartesian
from models import BiotNonlinearTpfa
from utility_functions import (plot_multiple_time_steps,
                               run_simulation_pairs_varying_parameters)

import porepy as pp


class SingleDimParameters:
    def _source_scalar(self, g: pp.Grid) -> np.ndarray:
        """Injection at domain center."""
        return (
            self._domain_center_source(g, val=self.params["source_value"])
            * g.cell_volumes
        )

    def _biot_alpha(self, g: pp.Grid) -> np.ndarray:
        """Injection at domain center."""
        return self.params["biot_alpha"] * np.ones(g.num_cells)

    def _bc_values_scalar(self, g: pp.Grid) -> np.ndarray:
        """Injection at domain center."""
        return 1e0 * np.ones(g.num_faces)

    def _reference_scalar(self, g: pp.Grid) -> np.ndarray:
        """Reference scalar value.

        Used for the scalar (pressure) contribution to stress.
        Parameters
        ----------
        g : pp.Grid
            Matrix grid.

        Returns
        -------
        np.ndarray
            Reference scalar value.

        """
        return 1e0 * np.ones(g.num_cells)

    def _initial_pressure(self, g):
        return 1e0 * np.ones(g.num_cells)


class Model(SingleDimParameters, BiotNonlinearTpfa):
    pass


if __name__ == "__main__":
    nc = 5
    params = {
        "time_step": 2e5,
        "end_time": 2e5,
        "plotting_file_name": "biot_matrix",
        "file_name": "biot_matrix",
        "grid_method": three_dimensional_cartesian,
        "n_cells": [nc, nc, nc],
        "mu": 10,
        "lambda": 10,
        "source_value": -3e1,
        "biot_alpha": 0.5,
        "n_time_steps": 1,
    }

    k = 1e-4
    update_params_lame = {
        "0": {"legend_title": "Lame parameters", "mu": k, "lambda": k},
        "2": {"mu": 1e2 * k, "lambda": 1e2 * k},
        "4": {"mu": 1e4 * k, "lambda": 1e4 * k},
        "6": {"mu": 1e6 * k, "lambda": 1e6 * k},
        "8": {"mu": 1e8 * k, "lambda": 1e8 * k},
    }
    k = 1e-2
    update_params_lame = {
        "0": {"legend_title": "Lame parameters", "mu": k, "lambda": k},
        "1": {"mu": 1e1 * k, "lambda": 1e1 * k},
        "2": {"mu": 1e2 * k, "lambda": 1e2 * k},
        "3": {"mu": 1e3 * k, "lambda": 1e3 * k},
        "4": {"mu": 1e4 * k, "lambda": 1e4 * k},
    }
    k = -0.02

    update_params = {
        "1": {"legend_title": "Source", "source_value": 8 * k},
        # "2": {"source_value": 4 * k},
        # "3": {"source_value": 2 * k},
        "4": {"source_value": 1 * k},
    }
    k = 0.1
    update_params = {
        "0.2": {"legend_title": "Biot alpha", "biot_alpha": 2 * k},
        "0.4": {"biot_alpha": 4 * k},
        "0.6": {"biot_alpha": 6 * k},
        "0.8": {"biot_alpha": 8 * k},
        "1.0": {"biot_alpha": 10 * k},
    }

    update_params = {
        "0.25": {"legend_title": "Biot alpha", "biot_alpha": 0.25},
        "0.50": {"biot_alpha": 0.5},
        "0.75": {"biot_alpha": 0.75},
        "1.00": {"biot_alpha": 1.0},
        # "1.0": {"biot_alpha": 10 * k},
    }

    run_simulation_pairs_varying_parameters(params, update_params, Model)
    for k in update_params.keys():
        update_params[k]["models"][0].params["plotting_file_name"] += "_linear"
    plot_multiple_time_steps(
        updates=update_params, n_steps=params["n_time_steps"], model_type="linear"
    )
