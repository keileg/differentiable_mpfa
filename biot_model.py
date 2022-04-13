import porepy as pp
import numpy as np
import scipy.sparse as sps
from incompressible_flow_nonlinear_tpfa import CommonModel
from plotting import plot_convergence
from grids import two_dimensional_cartesian, horizontal_fracture
from typing import List
from functools import partial
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class BiotNonlinearTpfa(CommonModel, pp.ContactMechanicsBiot):

    def _set_scalar_parameters(self) -> None:
        super()._set_scalar_parameters()
        self._bc_map = {}
        for g, d in self.gb:
            parameters = d[pp.PARAMETERS][self.scalar_parameter_key]
            self._bc_map[g] = (parameters["bc"])  #, parameters["bc_values"])



    def _permeability(self, g: pp.Grid):

        val = 1e-5 * np.ones(g.num_cells)

        if g.dim == self._Nd - 1:

            d = self.gb.node_props(g)
            data_edge = self.gb.edge_props((g, self._nd_grid()))
            projection = d["tangential_normal_projection"]
            u_local = self.reconstruct_local_displacement_jump(
                data_edge, projection
            )
            u_n = u_local[-1]
            val += u_n ** 2
        return val

    def _initial_pressure(self, g):
        return np.zeros(g.num_cells)

    def _initial_gap(self, g: pp.Grid) -> np.ndarray:
        return 1e-5 * np.ones(g.num_cells)

    def _permeability_function_ad(self, var: pp.ad.Ad_array):
        nc = self.gb.num_cells()
        val = pp.ad.Ad_array(1e-5 * np.ones(nc), sps.csr_matrix((nc, self.dof_manager.num_dofs()), dtype=float))
        fracture_subdomains: List[pp.Grid] = self.gb.grids_of_dimension(
            self._Nd - 1
        ).tolist()
        u_n = (self._ad.normal_component_frac * self._displacement_jump(fracture_subdomains)).evaluate(self.dof_manager)
        prol = self._ad.subdomain_projections_scalar.cell_prolongation(fracture_subdomains).evaluate(self.dof_manager)
        u_n_squared = u_n.copy()
        u_n_squared.jac.data = 2 * u_n_squared.jac.data
        u_n_squared.jac = u_n_squared.diagvec_mul_jac(u_n.val)
        u_n_squared.val = u_n.val ** 2
        val = (val + (prol * (u_n * u_n)))

        return val

    def _fluid_flux(self, subdomains: List[pp.Grid]) -> pp.ad.Operator:

        if self.params.get("use_tpfa", False) or self.gb.dim_max() == 1:
            base_discr = pp.ad.TpfaAd(self.scalar_parameter_key, subdomains)
        else:
            base_discr = pp.ad.MpfaAd(self.scalar_parameter_key, subdomains)
        self._ad.flux_discretization = base_discr


        TPFA_function = pp.ad.Function(
            partial(
                # For the moment, the actual implementation is a function differentiable_mpfa
                # located in pp.ad.discretizations. Both should be revised.
                pp.ad.differentiable_mpfa,
                grid_list=subdomains,
                bc=self._bc_map,
                base_discr=base_discr,
                dof_manager=self.dof_manager,
                var_name=self.scalar_variable,
                projections=pp.ad.SubdomainProjections(subdomains),
            ),
            "tpfa_ad",
        )
        p = self._ad.pressure
        u_j = self._ad.interface_displacement
        perm_function = pp.ad.Function(self._permeability_function_ad, "perm_function_biot")
        flux_ad = TPFA_function(perm_function, u_j, p)
        self._ad.dummy_eq_for_discretization = base_discr.flux * p

        bc_values = pp.ad.ParameterArray(
            self.scalar_parameter_key,
            array_keyword="bc_values",
            grids=subdomains,
        )
        vector_source_subdomains = pp.ad.ParameterArray(
            param_keyword=self.scalar_parameter_key,
            array_keyword="vector_source",
            grids=subdomains,
        )
        flux: pp.ad.Operator = (
                flux_ad
                + base_discr.bound_flux * bc_values
                + base_discr.bound_flux
                * self._ad.mortar_projections_scalar.mortar_to_primary_int
                * self._ad.interface_flux
                + base_discr.vector_source * vector_source_subdomains
        )
        if self.params.get("use_linear_discretization", False):
            return super()._fluid_flux(subdomains)
        return flux

    def _bc_values_scalar(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous Dirichlet values except at left end of fracture.
        """
        val = np.zeros(g.num_faces)
        if g.dim == 1:
            val[g.face_centers[0]<1e-5] = .0
        return val

    def _source_scalar(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous Dirichlet values except at left end of fracture.
        """
        val = np.zeros(g.num_cells)
        if g.dim == 1:
            val[g.cell_centers[0] < 3e-1] = .1
        return val

    def _aperture(self, g: pp.Grid) -> np.ndarray:
        """
        Aperture is a characteristic thickness of a cell, with units [m].
        1 in matrix, thickness of fractures and "side length" of cross-sectional
        area/volume (or "specific volume") for intersections of co-dimension 2 and 3.
        See also specific_volume.
        """
        aperture = np.ones(g.num_cells)
        # if g.dim < self._Nd:
        #     data_edge = self.gb.node_props()
        #     self.reconstruct_local_displacement_jump()
        return aperture


if __name__ == "__main__":
    num_iterations = 14
    params = {
        "use_tpfa": True,
        "linear_solver": "pypardiso",
        "max_iterations": num_iterations,
        "nl_convergence_tol": 1e-10,
        "nl_divergence_tol": 1e5,
        "grid_method": horizontal_fracture,
        "mesh_args": [2, 2],
        "use_ad": True,
        "time_step": 1e5,
        "end_time": 1e5,

    }
    nonlinear_model = BiotNonlinearTpfa(params)
    pp.run_time_dependent_model(nonlinear_model, params)
    params_linear = params.copy()
    params_linear["use_linear_discretization"] = True
    linear_model = BiotNonlinearTpfa(params_linear)
    pp.run_stationary_model(linear_model, params_linear)
    plot_convergence(nonlinear_model, linear_model, plot_errors=False)
    for m in [linear_model, nonlinear_model]:
        gf =m.gb.grids_of_dimension(1).tolist()

        u = m._displacement_jump(gf).evaluate(m.dof_manager)
        print(u.val)
        print("p", m._ad.pressure.evaluate(m.dof_manager).val)