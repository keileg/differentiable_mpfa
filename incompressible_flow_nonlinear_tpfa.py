"""
Script to develop a discretization of differentiable Tpfa/Mpfa.
"""
from typing import Any, List, Dict, Tuple
from functools import partial
import porepy as pp
import numpy as np
import scipy.sparse.linalg as spla

from pypardiso import spsolve as pardisosolve
import logging

from plotting import plot_convergence


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CommonModel:
    def __init__(self, params: Dict):
        super().__init__(params)
        self._solutions = []
        self._residuals = []

    def create_grid(self) -> None:
        """Create the grid bucket.

        A unit square grid with no fractures is assigned by default.

        The method assigns the following attributes to self:
            gb (pp.GridBucket): The produced grid bucket.
            box (dict): The bounding box of the domain, defined through minimum and
                maximum values in each dimension.
        """
        gb, box = self.params["grid_method"](self.params)
        self.box: Dict = box
        self.gb: pp.GridBucket = gb
        self.gb.compute_geometry()
        self._Nd = self.gb.dim_max()
        pp.contact_conditions.set_projections(self.gb)

    def _is_nonlinear_problem(self):
        return True

    def _flux(self, subdomains: List[pp.Grid]) -> pp.ad.Operator:
        try:
            key = self.scalar_parameter_key
            var = self.scalar_variable
        except:
            key = self.parameter_key
            var = self.variable
        if self.params.get("use_tpfa", False) or self.gb.dim_max() == 1:
            base_discr = pp.ad.TpfaAd(key, subdomains)
        else:
            base_discr = pp.ad.MpfaAd(key, subdomains)
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
                var_name=var,
                projections=pp.ad.SubdomainProjections(subdomains),
            ),
            "tpfa_ad",
        )
        p = self._ad.pressure
        perm_function = pp.ad.Function(self._permeability_function_ad, "perm_function")
        flux_ad = TPFA_function(perm_function, p, p)
        self._ad.dummy_eq_for_discretization = base_discr.flux * p

        bc_values = pp.ad.ParameterArray(
            key,
            array_keyword="bc_values",
            grids=subdomains,
        )
        vector_source_subdomains = pp.ad.ParameterArray(
            param_keyword=key,
            array_keyword="vector_source",
            grids=subdomains,
        )
        flux: pp.ad.Operator = (
            flux_ad
            + base_discr.bound_flux * bc_values
            + base_discr.bound_flux
            * self._ad.mortar_proj.mortar_to_primary_int
            * self._ad.mortar_flux
            + base_discr.vector_source * vector_source_subdomains
        )
        if self.params.get("use_linear_discretization", False):
            # super_model = pp.ContactMechanicsBiot if hasattr(self, "displacement_variable") else pp.IncompressibleFlow
            if hasattr(self, "displacement_variable"):
                return super()._fluid_flux(subdomains)
            else:
                return super()._flux(subdomains)
        return flux

    def before_newton_iteration(self) -> None:
        self._set_parameters()
        self._ad.dummy_eq_for_discretization.discretize(self.gb)

    def after_newton_iteration(self, solution_vector: np.ndarray) -> None:
        """
        Extract parts of the solution for current iterate.

        The iterate solutions in d[pp.STATE][pp.ITERATE] are updated for the
        mortar displacements and contact traction are updated.
        Method is a tailored copy from assembler.distribute_variable.

        Parameters:
            solution_vector (np.array): solution vector for the current iterate.

        """
        self._nonlinear_iteration += 1
        self.dof_manager.distribute_variable(
            values=solution_vector, additive=self._use_ad, to_iterate=True
        )

    def assemble_and_solve_linear_system(self, tol: float) -> np.ndarray:

        A, b = self._eq_manager.assemble()
        logger.debug(f"Max element in A {np.max(np.abs(A)):.2e}")
        logger.debug(
            f"Max {np.max(np.sum(np.abs(A), axis=1)):.2e} and min"
            + f" {np.min(np.sum(np.abs(A), axis=1)):.2e} A sum."
        )
        if self.linear_solver == "direct":
            dx = spla.spsolve(A, b)
        else:
            dx = pardisosolve(A, b)
            sol = dx + self.dof_manager.assemble_variable(from_iterate=True)
            if hasattr(self, "p_analytical"):
                error = np.linalg.norm(sol - self.p_analytical())
            else:
                error = 42
            logger.info(
                f"Error {error}, b {np.linalg.norm(b)} and dx {np.linalg.norm(dx)}"
            )
            self._solutions.append(sol)
        self._residuals.append(np.linalg.norm(b))
        return dx

    def check_convergence(
        self,
        solution: np.ndarray,
        prev_solution: np.ndarray,
        init_solution: np.ndarray,
        nl_params: Dict[str, Any],
    ) -> Tuple[float, bool, bool]:
        """Implements a convergence check, to be called by a non-linear solver.

        Parameters:
            solution (np.array): Newly obtained solution vector
            prev_solution (np.array): Solution obtained in the previous non-linear
                iteration.
            init_solution (np.array): Solution obtained from the previous time-step.
            nl_params (dict): Dictionary of parameters used for the convergence check.
                Which items are required will depend on the converegence test to be
                implemented.

        Returns:
            float: Error, computed to the norm in question.
            boolean: True if the solution is converged according to the test
                implemented by this method.
            boolean: True if the solution is diverged according to the test
                implemented by this method.

        Raises: NotImplementedError if the problem is nonlinear and AD is not used.
            Convergence criteria are more involved in this case, so we do not risk
            providing a general method.

        """

        # We normalize by the size of the solution vector
        error = self._residuals[-1]
        logger.debug(f"Normalized error: {error:.2e}")
        converged = error < nl_params["nl_convergence_tol"]
        diverged = False
        return error, converged, diverged

    def after_newton_failure(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        if self._is_nonlinear_problem():
            tol = self.params["nl_convergence_tol"]
            logger.info(f"Newton iterations did not converge to tolerance {tol}.")
        else:
            raise ValueError("Tried solving singular matrix for the linear problem.")

    def after_newton_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        super().after_newton_convergence(solution, errors, iteration_counter)
        self._export_step()

    def _export_step(self):
        self.exporter.write_vtu(data=["p"], time_dependent=True, time_step=1)


class NonlinearIncompressibleFlow(CommonModel, pp.IncompressibleFlow):
    def _set_parameters(self) -> None:
        super()._set_parameters()
        self._bc_map = {}
        for g, d in self.gb:
            parameters = d[pp.PARAMETERS][self.parameter_key]
            self._bc_map[g] = parameters["bc"]  # , parameters["bc_values"])

    def _permeability(self, g: pp.Grid):
        if hasattr(self, "dof_manager"):
            p = self.dof_manager.assemble_variable(
                [g], self.variable, from_iterate=True
            )[self.dof_manager.grid_and_variable_to_dofs(g, self.variable)]
        else:
            p = self._initial_pressure(g)
        return self._permeability_function(p)

    def _initial_pressure(self, g):
        return np.zeros(g.num_cells)

    def _permeability_function(self, var: np.ndarray):
        return np.ones(var.size)


if __name__ == "__main__":
    num_iterations = 15
    params = {
        "use_tpfa": True,
        "linear_solver": "pypardiso",
        "max_iterations": num_iterations,
        "nl_convergence_tol": 1e-10,
        "nl_divergence_tol": 1e5,
    }
    nonlinear_model = NonlinearIncompressibleFlow(params)
    pp.run_stationary_model(nonlinear_model, params)
    params_linear = params.copy()
    params_linear["use_linear_discretization"] = True
    linear_model = NonlinearIncompressibleFlow(params_linear)
    pp.run_stationary_model(linear_model, params_linear)
    plot_convergence(nonlinear_model, linear_model, plot_errors=True)
