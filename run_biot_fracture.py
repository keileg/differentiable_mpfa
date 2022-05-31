import numpy as np
from grids import horizontal_fracture_3d
from models import BiotNonlinearTpfa
from utility_functions import (plot_multiple_time_steps,
                               run_simulation_pairs_varying_parameters)

import porepy as pp


class MixedDimParameters:
    def _initial_gap(self, g: pp.Grid) -> np.ndarray:
        return 3e-2 * np.ones(g.num_cells)

    def _source_scalar(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous Dirichlet values except at left end of fracture."""
        val = np.zeros(g.num_cells)
        if g.dim == self._Nd - 1:
            val = self._domain_center_source(g, val=self.params["source_value"])
        return val

    def _biot_alpha(self, g: pp.Grid) -> np.ndarray:
        """Injection at domain center."""
        return self.params["biot_alpha"] * np.ones(g.num_cells)

    def after_newton_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        super().after_newton_convergence(solution, errors, iteration_counter)
        if self.time_index < self.params["n_time_steps"] - 1:
            self.time_step = 5e-1
        else:
            self.time_step = self.end_time - self.time + 1e-10
        self._ad.time_step._value = self.time_step


class Fracture(MixedDimParameters, BiotNonlinearTpfa):
    pass


if __name__ == "__main__":
    nc = 5
    params = {
        "use_tpfa": True,
        "time_step": 1e5,
        "end_time": 1e5,
        "plotting_file_name": "biot_fracture_temp",
        "file_name": "biot_fracture",
        "grid_method": horizontal_fracture_3d,
        "mesh_args": np.array([nc, nc, nc - 1]),
        "n_time_steps": 1,
        "biot_alpha": 0.2,
    }
    k = 1
    update_params = {
        "6": {"legend_title": "Source", "source_value": 6 * k},
        "4": {"source_value": 4 * k},
        "2": {"source_value": 2 * k},
        "1": {"source_value": 1 * k},
    }

    run_simulation_pairs_varying_parameters(params, update_params, Fracture)
    for k in update_params.keys():
        update_params[k]["models"][0].params["plotting_file_name"] += "_linear"
    plot_multiple_time_steps(
        updates=update_params, n_steps=params["n_time_steps"], model_type="linear"
    )
    # for i, name in enumerate(update_params.keys()):
    #     linear_model, m = update_params[name]["models"]
    #     jumps = m._displacement_jump(m.gb.grids_of_dimension(m._Nd - 1).tolist()).evaluate(m.dof_manager).val.reshape(
    #         (m._Nd, -1), order="F")
    #     print(name, jumps[-1])
    # sd = [m._nd_grid()]
    # phi = m._porosity_ad(sd).evaluate(m.dof_manager).val
    # print(max(phi), min(phi))
