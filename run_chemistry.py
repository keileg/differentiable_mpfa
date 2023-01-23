import logging

import numpy as np
import porepy as pp

from chemistry_model import Chemistry, ComponentConstants
from common_models import solid_values
from constitutive_laws import DomainCenterSource
from grids import three_dimensional_cartesian
from utility_functions import run_simulation_pairs_varying_parameters

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ChemistryParameters:
    def solute_sink(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Component a sink.

        Parameters:
            sd: Grid representing a subdomain.

        Returns:
            Cell-wise sink values.

        """
        assert len(subdomains) == 1
        sd = subdomains[0]
        rate = np.clip(self.params.get("fluid_source_value"), a_max=0, a_min=None)
        flow_vals = self.domain_center_source(sd, val=rate) * sd.cell_volumes
        sink = pp.wrap_as_ad_array(flow_vals, name="solute_sink")
        return sink

    def solute_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Component a source.

        Parameters:
            sd: Grid representing a subdomain.

        Returns:
            Cell-wise source values.

        """
        assert len(subdomains) == 1
        sd = subdomains[0]
        # Inject with a volume fraction half of the reference value.
        rate = (
            np.clip(self.params.get("fluid_source_value"), a_min=0, a_max=None)
            * self.component.solute_fraction()
            / 2
        )
        vals = self.domain_center_source(sd, val=rate) * sd.cell_volumes
        source = pp.wrap_as_ad_array(vals, name="solute_source")
        return source


class CombinedModel(ChemistryParameters, DomainCenterSource, Chemistry):
    pass


if __name__ == "__main__":
    nc = 15
    fluid = pp.FluidConstants({"compressibility": 0e0})
    # solid_values.update({"permeability": 1e1})
    solid = pp.SolidConstants(solid_values)
    params = {
        "use_tpfa": True,
        "grid_method": three_dimensional_cartesian,
        "plotting_file_name": "chemistry",
        "file_name": "chemistry",
        "folder_name": "chemistry",
        "time_manager": pp.TimeManager(schedule=[0, 1], dt_init=1, constant_dt=True),
        "n_cells": [nc, nc, nc],
        "compute_permeability_errors": False,
        "fluid_source_value": -1e3,
        "material_constants": {
            "fluid": fluid,
            "component": ComponentConstants(),
            "solid": solid,
        },
        "legend_title": r"Exponent $\eta$",
    }

    update_params = {
        "0": {"permeability_exponent": 0},
        "2": {"permeability_exponent": 2},
        "5": {"permeability_exponent": 5},
        "8": {"permeability_exponent": 8},
    }
    run_simulation_pairs_varying_parameters(params, update_params, CombinedModel)
