"""
Parameters and run script for Section 4.2.3
"""


import numpy as np
import porepy as pp

from grids import horizontal_fracture_3d
from models import BiotNonlinearTpfa
from utility_functions import (plot_multiple_time_steps,
                               run_simulation_pairs_varying_parameters)


class MixedDimParameters:
    def _initial_gap(self, sd: pp.Grid) -> np.ndarray:
        """Initial value for the fracture gap.

        Args:
            sd: Fracture subdomain grid.

        Returns:
            Cell-wise values.

        """
        return 2e-2 * np.ones(sd.num_cells)

    def _source_scalar(self, sd: pp.Grid) -> np.ndarray:
        """Injection at center of fracture.

        Args:
            sd: Subdomain grid.

        Returns:
            Cell-wise source value.

        """
        val = np.zeros(sd.num_cells)
        if sd.dim == self.nd - 1:
            val = self._domain_center_source(sd, val=self.params["source_value"])
        return val

    def _biot_alpha(self, sd: pp.Grid) -> np.ndarray:
        """Biot coefficient.

        Args:
            sd: Grid.

        Returns:
            Cell-wise Biot coefficient.

        """
        return self.params["biot_alpha"] * np.ones(sd.num_cells)


class Fracture(MixedDimParameters, BiotNonlinearTpfa):
    pass


if __name__ == "__main__":
    nc = 15
    params = {
        "use_tpfa": True,
        "time_manager": pp.TimeManager(
            schedule=[0, 1e5], dt_init=1e5, constant_dt=True
        ),
        "plotting_file_name": "biot_fracture",
        "file_name": "biot_fracture",
        "folder_name": "biot_fracture",
        "grid_method": horizontal_fracture_3d,
        "mesh_args": np.array([nc, nc, nc - 1]),
        "n_time_steps": 1,
        "biot_alpha": 0.2,
        "nl_convergence_tol": 1e-12,
    }
    k = 1e-1
    update_params = {
        "0.1": {"legend_title": r"Source [$m^3/s$]", "source_value": k},
        "0.2": {"source_value": 2 * k},
        "0.5": {"source_value": 5 * k},
        "1.0": {"source_value": 10 * k},
    }

    run_simulation_pairs_varying_parameters(params, update_params, Fracture)
    plot_multiple_time_steps(updates=update_params, n_steps=params["n_time_steps"])

    for k in update_params.keys():
        update_params[k]["models"][0].params["plotting_file_name"] += "_linear"
    plot_multiple_time_steps(
        updates=update_params, n_steps=params["n_time_steps"], model_type="linear"
    )
