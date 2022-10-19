import logging

import numpy as np
import porepy as pp

from grids import three_dimensional_cartesian
from models import Chemistry
from utility_functions import (plot_multiple_time_steps,
                               run_simulation_pairs_varying_parameters)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ChemistryParameters:
    def _set_chemistry_parameters(self) -> None:
        """Set parameters for chemical model.

        Returns:

        """
        for sd, data in self.mdg.subdomains(return_data=True):
            for comp in ["a", "c"]:
                parameters = {
                    "mass_weight": np.ones(sd.num_cells),
                    "density_inv": 1 / getattr(self, "_density_" + comp)(sd),
                }
                if comp == "c":
                    parameters.update(
                        {
                            "reference_porosity": self.rock.POROSITY
                            * np.ones(sd.num_cells),
                            "reference_reaction_rate": self._reference_reaction_rate(
                                sd
                            ),
                            "equilibrium_constant_inv": np.ones(sd.num_cells)
                            / self.params.get("equilibrium_constant", 0.1),
                        }
                    )

                else:
                    # Else covers only "a" but can be used for multiple dissolved components.
                    parameters.update(
                        {
                            "bc": getattr(self, "_bc_type_" + comp)(sd),
                            "bc_values": getattr(self, "_bc_values_" + comp)(sd),
                            "source": getattr(self, "_source_" + comp)(sd),
                            "stoichiometric_coefficient": getattr(
                                self, "_stoichiometric_coefficient_" + comp
                            )(sd),
                        }
                    )
                pp.initialize_data(
                    sd,
                    data,
                    getattr(self, comp + "_parameter_key"),
                    parameters,
                )

    def _bc_type_a(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Dirichlet conditions on all external boundaries.

        Returns:
            pp.BoundaryCondition:
        """
        # Define boundary regions
        all_bf, *_ = self._domain_boundary_sides(sd)
        # Define boundary condition on faces
        return pp.BoundaryCondition(sd, all_bf, "dir")

    def _bc_values_a(self, sd: pp.Grid) -> np.ndarray:
        """Homogeneous boundary values.

        Units:
            Dirichlet conditions: Pa = kg / m^1 / s^2
            Neumann conditions: m^3 / s
        """
        return 0.0 * np.ones(sd.num_faces)

    def _bc_values(self, sd: pp.Grid) -> np.ndarray:
        """Homogeneous boundary values.

        Units:
            Dirichlet conditions: Pa = kg / m^1 / s^2
            Neumann conditions: m^3 / s
        """
        vals = np.zeros(sd.num_faces)
        all_bf, *_ = self._domain_boundary_sides(sd)
        vals[all_bf] = 1e6
        return vals

    def _stoichiometric_coefficient_a(self, sd: pp.Grid) -> int:
        """Stoichiometric coefficient for the chemical reaction

        Args:
            sd: pp.Grid

        Returns:
            np.ndarray: (g.num_cells)

        """
        return self._alpha * np.ones(sd.num_cells)

    def _density_a(self, sd: pp.Grid) -> np.ndarray:
        """Density of the c component.

        Args:
            sd: pp.Grid

        Returns:
            np.ndarray: (g.num_cells)

        """
        return np.ones(sd.num_cells)

    def _density_c(self, sd: pp.Grid) -> np.ndarray:
        """Density of the c component.

        Args:
            sd: pp.Grid

        Returns:
            np.ndarray: (g.num_cells)

        """
        return self.params.get("density", 1) * np.ones(sd.num_cells)

    def _source_a(self, sd: pp.Grid) -> np.ndarray:
        """Component a source.

        Args:
            sd: Grid representing a subdomain.

        Returns:
            source_vals: array of cell-wise values.
        Homogeneous values except at the domain center.
        """
        rate = self.params.get("source_rate")
        source_vals = (
            self._domain_center_source(sd, val=0.0 * rate)
            # * sd.cell_volumes
            * self._density_a(sd)
        )
        return source_vals

    def _source(self, sd: pp.Grid) -> np.ndarray:
        """Flow source.

        Args:
            sd: Grid representing a subdomain.

        Returns:
            source_vals: array of cell-wise values.
        Homogeneous values except at the domain center.
        """
        rate = self.params.get("source_rate")
        source_vals = self._domain_center_source(sd, val=rate)  # * sd.cell_volumes
        return source_vals

    def _initial_a(self, sd: pp.Grid) -> np.ndarray:
        """Homogeneous initial values.

        Args:
            sd: pp.Grid of the subdomain.

        Returns:
            vals: np.ndarray of cell-wise concentration values
        """
        return 0.1 * np.ones(sd.num_cells)

    def _initial_c(self, sd: pp.Grid) -> np.ndarray:
        """Homogeneous initial values.

        Args:
            sd: pp.Grid of the subdomain.

        Returns:
            vals: np.ndarray of cell-wise precipitate values.

        Values should be in [0, reference_porosity], since the
        porosity is
            \phi = \phi_0 - c
        """
        return 0.04 * np.ones(sd.num_cells)

    def _initial_pressure(self, sd: pp.Grid) -> np.ndarray:
        """Homogeneous initial values.

        Args:
            sd: pp.Grid of the subdomain.

        Returns:
            vals: np.ndarray of cell-wise pressure values
        """
        return 1e6 * np.ones(sd.num_cells)

    def _reference_reaction_rate(self, sd: pp.Grid) -> np.ndarray:
        """Reference rate r_0 of the reaction term.

        Args:
            sd: pp.Grid representing a subdomain.

        Returns:
            r_0: array of cell-wise rates. Homogeneous values
                is the obvious choice.
        """
        r_0 = self.params["reaction_rate"] * np.ones(sd.num_cells)
        return r_0


class CombinedModel(ChemistryParameters, Chemistry):
    pass


if __name__ == "__main__":
    nc = 15
    params = {
        "use_tpfa": True,
        "grid_method": three_dimensional_cartesian,
        "plotting_file_name": "chemistry",
        "file_name": "chemistry",
        "folder_name": "chemistry",
        "time_manager": pp.TimeManager(schedule=[0, 1], dt_init=1, constant_dt=True),
        "n_cells": [nc, nc, nc],
        "compute_permeability_errors": False,
        "reaction_rate": 2e-4,
        "source_rate": -1e-4,
        "n_time_steps": 1,
    }

    update_params = {
        "0": {
            "legend_title": r"Exponent $\eta$",
            "permeability_exponent": 0,
        },
        "2": {"permeability_exponent": 2},
        "5": {"permeability_exponent": 5},
        "8": {"permeability_exponent": 8},
    }
    run_simulation_pairs_varying_parameters(params, update_params, CombinedModel)

    plot_multiple_time_steps(updates=update_params, n_steps=params["n_time_steps"])
    for k in update_params.keys():
        update_params[k]["models"][0].params["plotting_file_name"] += "_linear"
    plot_multiple_time_steps(
        updates=update_params, n_steps=params["n_time_steps"], model_type="linear"
    )
