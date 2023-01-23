"""
Parameters and run script for Section 4.2.2

Eight simulations are run: Four values for Biot coefficient, each run with TP and TPD.
"""
from typing import Any

import numpy as np
import porepy as pp

from common_models import Poromechanics, solid_values
from grids import three_dimensional_cartesian
from utility_functions import run_simulation_pairs_varying_parameters

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
        "fluid_source_value": 1e1,
    }

    # Biot coefficient varies:
    update_params: dict[str, Any] = {}
    biot_coefficients = [0.2, 0.4, 0.6, 0.8]
    solid_values.update(
        {
            "permeability": 1e-5,
        }
    )

    for i, val in enumerate(biot_coefficients):
        solid_values.update({"biot_coefficient": val})
        update_params[str(val)] = {
            "material_constants": {
                "solid": pp.SolidConstants(solid_values),
                "fluid": pp.FluidConstants({"compressibility": 1e-3}),
            },
            "legend_title": r"Biot coefficient $\alpha$",
        }

    run_simulation_pairs_varying_parameters(params, update_params, Poromechanics)
