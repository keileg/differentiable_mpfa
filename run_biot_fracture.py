"""
Parameters and run script for Section 4.2.3
"""


import numpy as np
import porepy as pp

from common_models import Poromechanics, solid_values
from grids import horizontal_fracture_3d
from utility_functions import run_simulation_pairs_varying_parameters

if __name__ == "__main__":
    nc = 15

    solid_values.update(
        {
            "permeability": 1e-5,
            "biot_coefficient": 0.2,
            "residual_aperture": 1e-1,
        }
    )
    solid = pp.SolidConstants(solid_values)
    fluid = pp.FluidConstants({"compressibility": 1e-3})
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
        "nl_convergence_tol": 1e-12,
        "material_constants": {"solid": solid, "fluid": fluid},
    }
    k = 1e1
    update_params = {
        "10": {"legend_title": r"Source [$m^3/s$]", "fluid_source_value": k},
        "20": {"fluid_source_value": 2 * k},
        "30": {"fluid_source_value": 3 * k},
        "40": {"fluid_source_value": 4 * k},
    }

    run_simulation_pairs_varying_parameters(params, update_params, Poromechanics)
    h = 1
