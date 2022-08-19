import numpy as np
from grids import three_dimensional_cartesian
from models import BiotNonlinearTpfa
from utility_functions import (
    plot_multiple_time_steps,
    run_simulation_pairs_varying_parameters,
)

import porepy as pp


class SingleDimParameters:
    def _source_scalar(self, sd: pp.Grid) -> np.ndarray:
        """Injection at domain center."""
        return (
            self._domain_center_source(sd, val=self.params["source_value"])
            * sd.cell_volumes
        )

    def _biot_alpha(self, sd: pp.Grid) -> np.ndarray:
        """Injection at domain center."""
        return self.params["biot_alpha"] * np.ones(sd.num_cells)

    def _bc_values_scalar(self, sd: pp.Grid) -> np.ndarray:
        """Injection at domain center."""
        return 1e0 * np.ones(sd.num_faces)

    def _reference_scalar(self, sd: pp.Grid) -> np.ndarray:
        """Reference scalar value.

        Used for the scalar (pressure) contribution to stress.
        Parameters
        ----------
        sd : pp.Grid
            Matrix grid.

        Returns
        -------
        np.ndarray
            Reference scalar value.

        """
        return 1e0 * np.ones(sd.num_cells)

    def _initial_pressure(self, sd):
        return 1e0 * np.ones(sd.num_cells)


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
        "nl_convergence_tol": 1e-12,
    }

    update_params = {
        "0.25": {"legend_title": r"Biot coefficient $\alpha$", "biot_alpha": 0.25},
        "0.50": {"biot_alpha": 0.5},
        "0.75": {"biot_alpha": 0.75},
        "1.00": {"biot_alpha": 1.0},
    }

    run_simulation_pairs_varying_parameters(params, update_params, Model)
    plot_multiple_time_steps(updates=update_params, n_steps=params["n_time_steps"])

    for k in update_params.keys():
        update_params[k]["models"][0].params["plotting_file_name"] += "_linear"
    plot_multiple_time_steps(
        updates=update_params, n_steps=params["n_time_steps"], model_type="linear"
    )
