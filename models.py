"""
Script to develop a discretization of differentiable Tpfa/Mpfa.
"""
import logging
from functools import partial
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.sparse as sps
from pypardiso import spsolve as pardisosolve
from utility_functions import (
    extract_line_solutions,
    load_converged_permeability,
    store_converged_permeability,
)

import porepy as pp

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Rock:
    """Mother of all rocks, all values are unity.

    Attributes:
        PERMEABILITY:
        POROSITY:
        LAMBDA: First Lame parameter
        MU: Second lame parameter / shear modulus
        YOUNG_MODULUS: Young's modulus
        POISSON_RATIO:

    """

    def __init__(self, mu, lmbda):
        self.PERMEABILITY = 1e-5
        self.DENSITY = 1
        self.POROSITY = 0.05
        self.MU = mu
        self.LAMBDA = lmbda
        self.YOUNG_MODULUS = pp.params.rock.young_from_lame(self.MU, self.LAMBDA)
        self.POISSON_RATIO = pp.params.rock.poisson_from_lame(self.MU, self.LAMBDA)


class CommonModel:
    def __init__(self, params: Dict):
        # Common solver parameters:
        default_params = {
            "linear_solver": "pypardiso",
            "max_iterations": 24,
            "nl_divergence_tol": 1e5,
            "use_ad": True,
            "nl_convergence_tol": 1e-12,
        }
        for key, val in default_params.items():
            if key not in params:
                params[key] = val
        super().__init__(params)
        self.rock = Rock(params.get("mu", 1), params.get("lambda", 1))
        self.fluid = params.get("fluid", pp.UnitFluid())
        self._solutions = []
        self._residuals = []
        self._permeability_errrors_nd = []
        self._permeability_errrors_frac = []
        self._export_times = []
        self.time = 0
        self.time_index = 0
        if "n_time_steps" in params:
            self.residual_list = []
            for _ in range(params["n_time_steps"]):
                self.residual_list.append([])

    def create_grid(self) -> None:
        """Create the mixed-dimensional grid.

        A unit square grid with no fractures is assigned by default.

        The method assigns the following attributes to self:
            mdg (pp.MixedDimensionalGrid): The produced mixed-dimensional grid.
            box (dict): The bounding box of the domain, defined through minimum and
                maximum values in each dimension.
        """
        mdg, box = self.params["grid_method"](self.params)
        self.box: Dict = box
        self.mdg: pp.MixedDimensionalGrid = mdg
        self.mdg.compute_geometry()
        self.nd = self.mdg.dim_max()
        pp.contact_conditions.set_projections(self.mdg)

    def _is_nonlinear_problem(self):
        return True

    def prepare_simulation(self):
        super().prepare_simulation()
        if self.params.get("compute_permeability_errors", False):
            load_converged_permeability(self)

    def _flux(self, subdomains: List[pp.Grid]) -> pp.ad.Operator:
        try:
            key = self.scalar_parameter_key
            var = self.scalar_variable
        except AttributeError:
            key = self.parameter_key
            var = self.variable
        if self.params.get("use_tpfa", False) or self.mdg.dim_max() == 1:
            base_discr = pp.ad.TpfaAd(key, subdomains)
        else:
            base_discr = pp.ad.MpfaAd(key, subdomains)
        self._ad.flux_discretization = base_discr
        p = self._ad.pressure

        flux_function = pp.ad.DifferentiableFVAd(
            subdomains=subdomains,
            mdg=self.mdg,
            base_discr=base_discr,
            dof_manager=self.dof_manager,
            permeability_function=self._permeability_function_ad,
            permeability_argument=getattr(self, "permeability_argument", p),
            potential=p,
            keyword=key,
        )

        # self._ad.flux_differentiated = flux_function.flux()
        self._ad.dummy_eq_for_discretization = base_discr.flux * p

        bc_values = pp.ad.ParameterArray(
            key,
            array_keyword="bc_values",
            subdomains=subdomains,
        )
        vector_source_subdomains = pp.ad.ParameterArray(
            param_keyword=key,
            array_keyword="vector_source",
            subdomains=subdomains,
        )
        flux: pp.ad.Operator = (
            flux_function.flux()
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
        self._ad.dummy_eq_for_discretization.discretize(self.mdg)

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
        for sd, data in self.mdg.subdomains(return_data=True):
            data[pp.STATE]["permeability"] = self._permeability(sd)
            perm_errors = np.inf
            do_compute = self.params.get("compute_permeability_errors", True)
            if do_compute and "converged_permeability" in data[pp.PARAMETERS]["flow"]:
                perm_errors = np.linalg.norm(
                    data[pp.PARAMETERS]["flow"]["converged_permeability"]
                    - data[pp.STATE]["permeability"]
                ) / np.sqrt(data[pp.STATE]["permeability"].size)
            if sd.dim == self.nd:
                self._permeability_errrors_nd.append(perm_errors)
            else:
                self._permeability_errrors_frac.append(perm_errors)
        extract_line_solutions(self)

    def assemble_and_solve_linear_system(self, tol: float) -> np.ndarray:

        A, b = self._eq_manager.assemble()
        logger.debug(f"Max element in A {np.max(np.abs(A)):.2e}")
        logger.debug(
            f"Max {np.max(np.sum(np.abs(A), axis=1)):.2e} and min"
            + f" {np.min(np.sum(np.abs(A), axis=1)):.2e} A sum."
        )
        if self.linear_solver == "direct":
            dx = sps.linalg.spsolve(A, b)
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
        res = np.linalg.norm(b)
        self._residuals.append(res)
        if hasattr(self, "residual_list"):
            self.residual_list[self.time_index - 1].append(res)
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
                Which items are required will depend on the convergence test to be
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
        self.exporter.write_vtu(data=self._export_names(), time_dependent=True)

    def after_simulation(self):
        self.exporter.write_pvd()
        self._set_parameters()
        store_converged_permeability(self)

    def _power_law_permeability(self, porosity):
        K_0 = self.rock.PERMEABILITY
        phi_0 = self.rock.POROSITY
        eta = self.params.get("permeability_exponent", 3)
        pbyp = porosity / phi_0
        perm = K_0 * pbyp**eta
        return perm

    def box_to_points(self) -> np.ndarray:
        """Obtain a bounding box for a point cloud.

        Returns:
            pts: np.ndarray (nd x 2). Point cloud. nd should be between 1 and 3

        """
        box = self.box
        pts = np.atleast_2d([[box["xmin"], box["xmax"]]])
        if "ymin" in box:
            pts = np.vstack((pts, np.atleast_2d([[box["ymin"], box["ymax"]]])))
        if "zmin" in box:
            pts = np.vstack((pts, np.atleast_2d([[box["zmin"], box["zmax"]]])))
        return pts

    def _domain_center_source(self, sd: pp.Grid, val: float) -> np.ndarray:
        vals = np.zeros(sd.num_cells)
        # if g.dim == self._Nd - 1:
        domain_corners = self.box_to_points()

        pt = np.zeros((3, 1))
        # Compute domain center
        pt[: self.nd] = np.reshape(np.mean(domain_corners, axis=1), (self.nd, 1))
        # Translate a tiny distance to make determination unique in case of even
        # number of Cartesian cells
        pt -= 1e-10
        ind = sd.closest_cell(pt)
        vals[ind[0]] = val
        return vals


class NonlinearIncompressibleFlow(CommonModel, pp.IncompressibleFlow):
    def _export_names(self):
        return [self.variable]

    def _permeability(self, sd: pp.Grid):
        p = self.dof_manager.assemble_variable([sd], self.variable, from_iterate=True)[
            self.dof_manager.grid_and_variable_to_dofs(sd, self.variable)
        ]
        return self._permeability_function(p)


class Chemistry(NonlinearIncompressibleFlow):
    def __init__(self, params: Dict):
        super().__init__(params)
        self.time_step = params["time_step"]
        self.end_time = params["end_time"]
        self.component_a_variable = "A"
        self.component_c_variable = "C"
        self.a_parameter_key = "chemistry_parameters_a"
        self.c_parameter_key = "chemistry_parameters_c"
        # Stoichiometric constant
        self._alpha = self.params.get("alpha", 1)
        self._equilibrium_constant = self.params.get("equilibrium_constant", 0.5)

    def _set_parameters(self) -> None:
        """
        Assigns chemistry and flow parameters to the data dictionaries of the subdomains
        present in self.gb.

        Order could be important, depending on implementation - flow parameters might depend on chemistry.
        """

        self._set_chemistry_parameters()
        super()._set_parameters()

    def _initial_condition(self):
        """Initial condition for all primary variables and the parameter representing the
        secondary flux variable (needs to be present for first call to upwind discretization).
        """
        for sd, data in self.mdg.subdomains(return_data=True):
            state = {
                self.component_a_variable: self._initial_a(sd),
                self.component_c_variable: self._initial_c(sd),
                self.variable: self._initial_pressure(sd),
            }
            data[pp.STATE] = state
            data[pp.STATE][pp.ITERATE] = state.copy()
            pp.initialize_data(
                sd, data, self.a_parameter_key, {"darcy_flux": np.zeros(sd.num_faces)}
            )

    def _assign_variables(self) -> None:
        """
        Assign primary variables to the subdomains and interfaces of the mixed-dimensional grid.
        """
        super()._assign_variables()
        # First for the subdomains
        for sd, data in self.mdg.subdomains(return_data=True):
            data[pp.PRIMARY_VARIABLES].update(
                {
                    self.component_a_variable: {"cells": 1},
                    self.component_c_variable: {"cells": 1},
                }
            )

    def _create_ad_variables(self) -> None:
        """Create the merged variables for potential and mortar flux"""
        pp.IncompressibleFlow._create_ad_variables(self)
        self._ad.component_a = self._eq_manager.merge_variables(
            [(sd, self.component_a_variable) for sd in self.mdg.subdomains()]
        )
        self._ad.component_c = self._eq_manager.merge_variables(
            [(sd, self.component_c_variable) for sd in self.mdg.subdomains()]
        )
        self.permeability_argument = self._ad.component_c

    def _permeability(self, sd: pp.Grid):
        phi = self._porosity(sd)
        return self._power_law_permeability(phi)

    def _porosity(self, sd):
        params = self.mdg.subdomain_data(sd)[pp.PARAMETERS][self.c_parameter_key]
        rho_c = params["density_inv"]
        phi_0 = params["reference_porosity"]

        # Is this the wanted behaviour of assemble_variable, EK?
        concentration_c = self.dof_manager.assemble_variable(
            [sd], self.component_c_variable, from_iterate=True
        )[self.dof_manager.grid_and_variable_to_dofs(sd, self.component_c_variable)]
        # TODO: fix ad division
        phi = phi_0 * (1 - concentration_c * rho_c)
        return phi

    def _permeability_function_ad(self, concentration_c: pp.ad.Ad_array):
        subdomains = self.mdg.subdomains()

        rho_c = pp.ad.ParameterMatrix(
            self.c_parameter_key,
            "density_inv",
            subdomains,
            name="rho_0",
        ).evaluate(self.dof_manager)
        phi_0m = pp.ad.ParameterMatrix(
            self.c_parameter_key,
            "reference_porosity",
            subdomains,
            name="phi_0",
        ).parse(self.mdg)
        phi_0a = pp.ad.ParameterArray(
            self.c_parameter_key,
            "reference_porosity",
            subdomains,
            name="phi_0",
        ).parse(self.mdg)
        # phi_0 = pp.ad.Ad_array(phi_0, sps.csr_matrix(concentration_c.jac.shape))
        phi = (-1) * phi_0m * (
            rho_c * concentration_c
        ) + phi_0a  # self._porosity_ad(concentration_c, rho_c, phi_0)
        permeability = self._power_law_permeability(phi)
        return permeability

    def _porosity_ad(self, concentration_c, rho_c, phi_0):
        # return phi_0 * (1 - rho_c * concentration_c)
        subdomains = self.mdg.subdomains()

        phi_0a = pp.ad.ParameterArray(
            self.c_parameter_key,
            "reference_porosity",
            subdomains,
            name="phi_0",
        )
        val = (-1) * phi_0 * (rho_c * concentration_c) + phi_0a
        # val = phi_0a - concentration_c
        return val

    def _advective_flux(
        self, subdomains: List[pp.Grid], component: str
    ) -> pp.ad.Operator:
        """

        Args:
            subdomains: List of subdomain grids, assumed to contain a single entry.
            component: Name of the advected component. Not hardcoded to "a" to facilitate
                multi-component extension.


        Returns:
            advection: pp.ad.Operator representing the advection term.

        Assigns an Operator as an attribute to self._ad. It is used for (re)computation of the
        Upwind discretization.
        """
        key = getattr(self, component + "_parameter_key")
        var = getattr(self._ad, "component_" + component)

        bc_val = pp.ad.BoundaryCondition(key, subdomains)

        upwind = pp.ad.UpwindAd(key, subdomains)
        setattr(
            self._ad,
            "dummy_upwind_for_discretization_" + component,
            upwind.upwind * var,
        )
        flux = self._flux(subdomains)
        advection = (
            flux * (upwind.upwind * var)
            - upwind.bound_transport_dir * (flux * bc_val)
            - upwind.bound_transport_neu * bc_val
        )
        return advection

    def _assign_equations(self) -> None:
        """Define equations through discretizations."""

        self._assign_flow_equations()
        subdomains = self._ad.subdomains
        transport_equation_a: pp.ad.Operator = self._subdomain_transport_equation(
            subdomains, component="a"
        )
        mass_equation_c: pp.ad.Operator = self._precipitate_mass_equation(
            subdomains, dissolved_components=["a"]
        )
        # Assign equations to manager
        self._eq_manager.name_and_assign_equations(
            {
                "subdomain_transport_a": transport_equation_a,
                "subdomain_mass_c": mass_equation_c,
            },
        )

    def _assign_flow_equations(self):
        NonlinearIncompressibleFlow._assign_equations(self)
        self._ad.time_step = pp.ad.Scalar(self.time_step, "time_step")

        subdomains = self._ad.subdomains

        rho_c = pp.ad.ParameterMatrix(
            self.c_parameter_key,
            "density_inv",
            subdomains,
            name="rho_0",
        )
        phi_0 = pp.ad.ParameterMatrix(
            self.c_parameter_key,
            "reference_porosity",
            subdomains,
            name="phi_0",
        )
        var_c = self._ad.component_c
        d_phi = self._porosity_ad(var_c, rho_c, phi_0) - self._porosity_ad(var_c.previous_timestep(), rho_c, phi_0)
        accumulation_term = d_phi / self._ad.time_step
        # self._eq_manager.equations["subdomain_flow"] += accumulation_term

    def _precipitate_mass_equation(
        self, subdomains: List[pp.Grid], dissolved_components
    ) -> pp.ad.Operator:
        """Precipitate mass balance equation.


        Parameters
        ----------
        subdomains : List[pp.Grid]
            Subdomains on which the equation is defined.

        Returns
        -------
        eq : pp.ad.Operator
            The equation on AD form.

        """
        key = self.c_parameter_key
        var_c = self._ad.component_c
        mass_discr = pp.ad.MassMatrixAd(key, subdomains)

        # Flow parameters
        rho_c = pp.ad.ParameterMatrix(
            self.c_parameter_key,
            "density_inv",
            subdomains,
            name="rho_0",
        )
        phi_0 = pp.ad.ParameterMatrix(
            self.c_parameter_key,
            "reference_porosity",
            subdomains,
            name="phi_0",
        )

        # Mass matrix to integrate over cell
        c_increment = mass_discr.mass * (var_c - var_c.previous_timestep())
        accumulation = (
            self._porosity_ad(var_c, rho_c, phi_0) * c_increment
        ) / self._ad.time_step

        reaction_term = mass_discr.mass * self._reaction_term(
            subdomains, dissolved_components
        )
        eq = accumulation - reaction_term
        return eq

    def _reaction_term(
        self, subdomains: List[pp.Grid], dissolved_components: List[str]
    ) -> pp.ad.Operator:
        """Reaction term.


        Parameters
        ----------
        subdomains : List[pp.Grid]
            Subdomains on which the equation is defined.

        Returns
        -------
        eq : pp.ad.Operator
            Reaction term on AD form.

        """
        rho_c_inv = pp.ad.ParameterMatrix(
            self.c_parameter_key,
            "density_inv",
            subdomains,
            name="rho_C",
        )
        phi_0 = pp.ad.ParameterMatrix(
            self.c_parameter_key,
            "reference_porosity",
            subdomains,
            name="phi_0",
        )
        r_0 = pp.ad.ParameterMatrix(
            self.c_parameter_key,
            "reference_reaction_rate",
            subdomains,
            name="r_0",
        )
        equilibrium_constant_inv = pp.ad.ParameterMatrix(
            param_keyword=self.c_parameter_key,
            array_keyword="equilibrium_constant_inv",
            subdomains=subdomains,
        )
        product = 1
        for component in dissolved_components:
            var = getattr(self._ad, "component_" + component)
            rho_var_inv = pp.ad.ParameterMatrix(
                getattr(self, component + "_parameter_key"),
                "density_inv",
                subdomains,
                name="rho_C",
            )
            product *= rho_var_inv * var
        # product *= (1-self._ad.component_c)
        phi = self._porosity_ad(self._ad.component_c, rho_c_inv, phi_0)
        rate = self._area_factor(phi, subdomains) * (
            r_0 * (equilibrium_constant_inv * product - 1)
        )
        return rate

    def _area_factor(self, phi, subdomains):
        phi_0 = pp.ad.ParameterArray(
            self.c_parameter_key,
            "reference_porosity",
            subdomains,
            name="phi_0",
        )
        return phi_0 + 0.1 * (phi - phi_0)

    def _subdomain_transport_equation(
        self, subdomains: List[pp.Grid], component: str
    ) -> pp.ad.Operator:
        """Advective transport equations.


        Parameters
        ----------
        subdomains : List[pp.Grid]
            Subdomains on which the equation is defined.

        Returns
        -------
        eq : pp.ad.Operator
            The equation on AD form.

        """
        key = getattr(self, component + "_parameter_key")
        var = getattr(self._ad, "component_" + component)
        var_c = self._ad.component_c
        mass_discr = pp.ad.MassMatrixAd(key, subdomains)

        # Flow parameters

        source = pp.ad.ParameterArray(
            param_keyword=key,
            array_keyword="source",
            subdomains=subdomains,
        )

        rho_c = pp.ad.ParameterMatrix(
            self.c_parameter_key,
            "density_inv",
            subdomains,
            name="rho_0",
        )
        phi_0 = pp.ad.ParameterMatrix(
            self.c_parameter_key,
            "reference_porosity",
            subdomains,
            name="phi_0",
        )

        div = pp.ad.Divergence(subdomains=subdomains)

        # Accumulation term on the fractures.
        accumulation = (
            self._porosity_ad(var_c, rho_c, phi_0)
            * (mass_discr.mass * (var - var.previous_timestep()))
        ) / self._ad.time_step

        advection = self._advective_flux(subdomains, component)
        # Mass matrix to integrate over cell
        reaction_term = mass_discr.mass * self._reaction_term(subdomains, [component])
        eq = accumulation + div * advection - source + reaction_term
        return eq

    def before_newton_iteration(self):
        super().before_newton_iteration()
        self.compute_fluxes()
        self._ad.dummy_upwind_for_discretization_a.discretize(self.mdg)

    def compute_fluxes(self) -> None:
        pp.fvutils.compute_darcy_flux(
            self.mdg,
            keyword=self.parameter_key,
            keyword_store=self.a_parameter_key,
            d_name="darcy_flux",
            p_name=self.variable,
            from_iterate=True,
        )

    def _export_names(self):
        return [
            self.variable,
            self.component_a_variable,
            self.component_c_variable,
            "permeability",
        ]


class BiotNonlinearTpfa(CommonModel, pp.ContactMechanicsBiot):
    def __init__(self, params):
        super().__init__(params)

        self.time_index = 0
        self._nonlinear_iteration = 0

    def _set_scalar_parameters(self) -> None:
        super()._set_scalar_parameters()
        self._bc_map = {}

        for sd, data in self.mdg.subdomains(return_data=True):
            specific_volume = self._specific_volume(sd)

            comp = self.fluid.COMPRESSIBILITY
            porosity = self._porosity(sd)
            phi_0 = self.rock.POROSITY * np.ones(sd.num_cells)
            alpha = self._biot_alpha(sd)
            bulk = pp.params.rock.bulk_from_lame(self.rock.LAMBDA, self.rock.MU)
            new_params = {}
            if sd.dim == self.nd:
                mass_weight = (
                    porosity * comp + (alpha - phi_0) / bulk
                ) * self.scalar_scale
                new_params.update(
                    {
                        "N_inverse": (self._biot_alpha(sd) - phi_0) / bulk,
                        "phi_reference": phi_0,
                    }
                )
            else:
                mass_weight = porosity * comp * self.scalar_scale
            new_params.update({"mass_weight": mass_weight * specific_volume})
            parameters = data[pp.PARAMETERS][self.scalar_parameter_key]
            parameters.update(new_params)
            self._bc_map[sd] = {
                "type": parameters["bc"],
                "values": parameters["bc_values"],
            }

    def _stiffness_tensor(self, sd: pp.Grid) -> pp.FourthOrderTensor:
        """Fourth order stress tensor.


        Parameters
        ----------
        sd : pp.Grid
            Matrix grid.

        Returns
        -------
        pp.FourthOrderTensor
            Cell-wise representation of the stress tensor.

        """
        lam = self.rock.LAMBDA * np.ones(sd.num_cells)
        mu = self.rock.MU * np.ones(sd.num_cells)
        return pp.FourthOrderTensor(mu, lam)

    def _permeability(self, sd: pp.Grid):
        if sd.dim == self.nd:
            phi = self._porosity(sd)
            vals = self._power_law_permeability(phi)
        else:
            vals = self._aperture(sd) ** 2
        return vals

    def _initial_pressure(self, sd):
        return np.zeros(sd.num_cells)

    def _porosity_ad(self, subdomain):
        # Matrix porosity
        restriction = self._ad.subdomain_projections_scalar.cell_restriction(subdomain)
        p = restriction * self._ad.pressure
        one_by_n = pp.ad.ParameterMatrix(
            param_keyword=self.scalar_parameter_key,
            array_keyword="N_inverse",
            subdomains=subdomain,
        )
        p_0 = pp.ad.ParameterArray(
            param_keyword=self.mechanics_parameter_key,
            array_keyword="p_reference",
            subdomains=subdomain,
        )
        phi_0 = pp.ad.ParameterArray(
            param_keyword=self.scalar_parameter_key,
            array_keyword="phi_reference",
            subdomains=subdomain,
        )
        phi = phi_0 + self._div_u(subdomain) + one_by_n * (p - p_0)
        return phi

    def _porosity(self, sd):
        if sd.dim < self.nd:
            vals = np.ones(sd.num_cells)
        elif self.time_index == 0 and self._nonlinear_iteration == 0:
            vals = self.rock.POROSITY * np.ones(sd.num_cells)
        else:
            vals = self._porosity_ad([sd]).evaluate(self.dof_manager).val
        return vals

    def _permeability_function_ad(self, var: pp.ad.Ad_array):
        phi = self._porosity_ad([self._nd_subdomain()]).evaluate(self.dof_manager)
        fracture_subdomains: List[pp.Grid] = self.mdg.subdomains(dim=self.nd - 1)
        u_n = self._ad.normal_component_frac * self._displacement_jump(
            fracture_subdomains
        )
        max_ad = pp.ad.Function(pp.ad.functions.maximum, "maximum")
        u_n = max_ad(u_n, self._gap(fracture_subdomains)).evaluate(self.dof_manager)
        prolongation_frac = self._ad.subdomain_projections_scalar.cell_prolongation(
            fracture_subdomains
        ).evaluate(self.dof_manager)
        prolongation_matrix = self._ad.subdomain_projections_scalar.cell_prolongation(
            [self._nd_subdomain()]
        ).evaluate(self.dof_manager)
        frac = u_n * u_n * u_n
        val = (
            prolongation_frac * frac
            + prolongation_matrix * self._power_law_permeability(phi)
        )
        return val

    def _fluid_flux(self, subdomains: List[pp.Grid]) -> pp.ad.Operator:

        if self.params.get("use_tpfa", False) or self.mdg.dim_max() == 1:
            base_discr = pp.ad.TpfaAd(self.scalar_parameter_key, subdomains)
        else:
            base_discr = pp.ad.MpfaAd(self.scalar_parameter_key, subdomains)
        self._ad.flux_discretization = base_discr

        p = self._ad.pressure
        differentiated_tpfa = pp.ad.DifferentiableFVAd(
            subdomains=subdomains,
            mdg=self.mdg,
            base_discr=base_discr,
            dof_manager=self.dof_manager,
            permeability_function=self._permeability_function_ad,
            permeability_argument=self._ad.interface_displacement,
            potential=p,
            keyword=self.scalar_parameter_key,
        )
        self._ad.dummy_eq_for_discretization = base_discr.flux * p
        self._ad.differentiated_tpfa = differentiated_tpfa
        bc_values = pp.ad.ParameterArray(
            self.scalar_parameter_key,
            array_keyword="bc_values",
            subdomains=subdomains,
        )
        vector_source_subdomains = pp.ad.ParameterArray(
            param_keyword=self.scalar_parameter_key,
            array_keyword="vector_source",
            subdomains=subdomains,
        )
        flux: pp.ad.Operator = (
            differentiated_tpfa.flux()
            + base_discr.bound_flux * bc_values
            + base_discr.bound_flux
            * self._ad.mortar_projections_scalar.mortar_to_primary_int
            * self._ad.interface_flux
            + base_discr.vector_source * vector_source_subdomains
        )
        if self.params.get("use_linear_discretization", False):
            return super()._fluid_flux(subdomains)
        return flux

    def _boundary_pressure(self, subdomains: List[pp.Grid]) -> pp.ad.Operator:
        if self.params.get("use_linear_discretization", False):
            return super()._boundary_pressure(subdomains)

        flux_discr = self._ad.flux_discretization
        bc = pp.ad.ParameterArray(
            self.scalar_parameter_key,
            array_keyword="bc_values",
            subdomains=subdomains,
        )

        vector_source_subdomains = pp.ad.ParameterArray(
            param_keyword=self.scalar_parameter_key,
            array_keyword="vector_source",
            subdomains=subdomains,
        )
        p_primary = (
            flux_discr.bound_pressure_cell * self._ad.pressure
            + self._ad.differentiated_tpfa.bound_pressure_face()
            * (
                self._ad.mortar_projections_scalar.mortar_to_primary_int
                * self._ad.interface_flux
            )
            + self._ad.differentiated_tpfa.bound_pressure_face() * bc
            + flux_discr.vector_source * vector_source_subdomains
        )
        return p_primary

    def _aperture(self, sd: pp.Grid) -> np.ndarray:
        """
        Aperture is a characteristic thickness of a cell, with units [m].
        1 in matrix, thickness of fractures and "side length" of cross-sectional
        area/volume (or "specific volume") for intersections of co-dimension 2 and 3.
        See also specific_volume.
        """
        aperture = np.ones(sd.num_cells)
        if sd.dim < self.nd:
            intf = self.mdg.subdomain_pair_to_interface((sd, self._nd_subdomain()))
            proj = self.mdg.subdomain_data(sd)["tangential_normal_projection"]
            u_j = self.reconstruct_local_displacement_jump(intf, proj)
            aperture = np.max(np.vstack((u_j[-1], self._initial_gap(sd))), axis=0)
        return aperture

    def _initial_condition(self):
        """Initial condition for all primary variables and the parameter representing the
        secondary flux variable (needs to be present for first call to upwind discretization).
        """
        super()._initial_condition()
        for sd, data in self.mdg.subdomains(return_data=True):
            state = {
                self.scalar_variable: self._initial_pressure(sd),
            }
            data[pp.STATE].update(state)
            data[pp.STATE][pp.ITERATE].update(state.copy())

    def after_newton_iteration(self, solution_vector: np.ndarray) -> None:
        """Prepare for plotting of displacements.

        This is done after each iteration in order to facilitate export of
        non-converged solutions.

        Parameters:
            solution_vector (np.array): solution vector for the current iterate.

        """
        super().after_newton_iteration(solution_vector)
        nd = self.nd
        for sd, data in self.mdg.subdomains(return_data=True):
            displacements = np.zeros((3, sd.num_cells))
            if sd.dim == nd:
                displacements[:nd] = data[pp.STATE][pp.ITERATE][
                    self.displacement_variable
                ].reshape((nd, sd.num_cells), order="F")

            data[pp.STATE]["displacements"] = displacements
            data[pp.STATE]["aperture"] = self._aperture(sd)

    def _export_names(self):
        return [self.scalar_variable, "displacements", "aperture", "permeability"]
