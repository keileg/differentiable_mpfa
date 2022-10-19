"""
Parameters and run script for Section 4.2.2

Eight simulations are run: Four values for Biot coefficient, each run with TP and TPD.
"""
import numpy as np
import porepy as pp

from grids import three_dimensional_cartesian
from models import BiotNonlinearTpfa
from utility_functions import (plot_multiple_time_steps,
                               run_simulation_pairs_varying_parameters)


class SingleDimParameters:
    def _source_scalar(self, sd: pp.Grid) -> np.ndarray:
        """Injection at domain center.

        Args:
            sd: Subdomain grid.

        Returns:
            Cell-wise source value.

        """
        return self._domain_center_source(sd, val=1e-1)

    def _biot_alpha(self, sd: pp.Grid) -> np.ndarray:
        """Biot coefficient.

        Args:
            sd: Grid.

        Returns:
            Cell-wise Biot coefficient.

        """
        return self.params["biot_alpha"] * np.ones(sd.num_cells)


class Model(SingleDimParameters, BiotNonlinearTpfa):
    """Combine parameters and model."""

    pass


if __name__ == "__main__":
    nc = 15
    # Parameters used for all simulations
    params = {
        "use_tpfa": True,
        "time_manager": pp.TimeManager(
            schedule=[0, 1e5], dt_init=1e5, constant_dt=True
        ),
        "plotting_file_name": "biot_matrix",
        "file_name": "biot_matrix",
        "folder_name": "biot_matrix",
        "grid_method": three_dimensional_cartesian,
        "n_cells": [nc, nc, nc],
        "n_time_steps": 1,
        "nl_convergence_tol": 1e-12,
    }

    # Biot coefficient varies:
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
