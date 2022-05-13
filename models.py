"""
Script to develop a discretization of differentiable Tpfa/Mpfa.
"""
from typing import Any, List, Dict, Tuple
from functools import partial
import porepy as pp
import numpy as np
import scipy.sparse as sps

from pypardiso import spsolve as pardisosolve
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Rock(object):
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
        self.POROSITY = .05
        self.MU = mu
        self.LAMBDA = lmbda
        self.YOUNG_MODULUS = pp.params.rock.young_from_lame(self.MU, self.LAMBDA)
        self.POISSON_RATIO = pp.params.rock.poisson_from_lame(self.MU, self.LAMBDA)


class CommonModel:
    def __init__(self, params: Dict):
        # Common solver parameters:
        params.update({
            "linear_solver": "pypardiso",
            "max_iterations": 15,
            "nl_convergence_tol": 1e-12,
            "nl_divergence_tol": 1e5,
            "use_ad": True,
        })
        super().__init__(params)
        self.rock = Rock(params.get("mu", 1), params.get("lambda", 1))
        self.fluid = params.get("fluid", pp.UnitFluid())
        self._solutions = []
        self._residuals = []
        self._export_times = []
        self.time = 0
        self.time_index = 0
        if "n_time_steps" in params:
            self.residual_list = []
            for _ in range(params["n_time_steps"]):
                self.residual_list.append([])

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
        except AttributeError:
            key = self.parameter_key
            var = self.variable
        if self.params.get("use_tpfa", False) or self.gb.dim_max() == 1:
            base_discr = pp.ad.TpfaAd(key, subdomains)
        else:
            base_discr = pp.ad.MpfaAd(key, subdomains)
        self._ad.flux_discretization = base_discr

        flux_function = pp.ad.DifferentiableFVAd(
            pp.ad.differentiable_mpfa,
            grid_list=subdomains,
            bc=self._bc_map,
            base_discr=base_discr,
            dof_manager=self.dof_manager,
            var_name=var,
        )

        p = self._ad.pressure
        perm_function = pp.ad.Function(self._permeability_function_ad, "perm_function")
        perm_argument = getattr(self, "permeability_argument", p)
        flux_ad = flux_function(perm_function, perm_argument, p)
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

        for g, d in self.gb:
            d[pp.STATE]["permeability"] = self._permeability(g)

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
        self.exporter.write_vtu(data=self._export_names(), time_dependent=True)

    def after_simulation(self):
        self.exporter.write_pvd(self.exporter._exported_time_step_file_names)

    def _kozeny_carman(self, porosity):
        K_0 = self.rock.PERMEABILITY
        phi_0 = self.rock.POROSITY
        pbyp = porosity / phi_0
        perm = K_0 * pbyp * pbyp * pbyp
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

    def _domain_center_source(self, g: pp.Grid, val: float) -> np.ndarray:
        vals = np.zeros(g.num_cells)
        # if g.dim == self._Nd - 1:
        domain_corners = self.box_to_points()

        pt = np.zeros((3, 1))
        # Compute domain center
        pt[:self._Nd] = np.reshape(np.mean(domain_corners, axis=1), (self._Nd, 1))
        # Translate a tiny distance to make determination unique in case of even
        # number of Cartesian cells
        pt -= 1e-10
        ind = g.closest_cell(pt)
        vals[ind[0]] = val
        return vals


class NonlinearIncompressibleFlow(CommonModel, pp.IncompressibleFlow):
    def _export_names(self):
        return [self.variable]

    def _set_parameters(self) -> None:
        super()._set_parameters()
        self._bc_map = {}
        for g, d in self.gb:
            parameters = d[pp.PARAMETERS][self.parameter_key]
            self._bc_map[g] = {"type": parameters["bc"],
                               "values": parameters["bc_values"],
                               }

    def _permeability(self, g: pp.Grid):
        p = self.dof_manager.assemble_variable(
            [g], self.variable, from_iterate=True
        )[self.dof_manager.grid_and_variable_to_dofs(g, self.variable)]
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
        self._equilibrium_constant = self.params.get("equilibrium_constant", .5)

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
        for g, d in self.gb:
            state = {
                self.component_a_variable: self._initial_a(g),
                self.component_c_variable: self._initial_c(g),
                self.variable: self._initial_pressure(g),
            }
            d[pp.STATE] = state
            d[pp.STATE][pp.ITERATE] = state.copy()
            pp.initialize_data(g, d, self.a_parameter_key, {"darcy_flux": np.zeros(g.num_faces)})

    def _assign_variables(self) -> None:
        """
        Assign primary variables to the nodes and edges of the grid bucket.
        """
        super()._assign_variables()
        # First for the nodes
        for g, d in self.gb:
            d[pp.PRIMARY_VARIABLES].update({self.component_a_variable: {"cells": 1},
                                            self.component_c_variable: {"cells": 1},
                                            })

    def _create_ad_variables(self) -> None:
        """Create the merged variables for potential and mortar flux"""
        pp.IncompressibleFlow._create_ad_variables(self)
        grid_list = [g for g, _ in self.gb.nodes()]
        self._ad.component_a = self._eq_manager.merge_variables(
            [(g, self.component_a_variable) for g in grid_list]
        )
        self._ad.component_c = self._eq_manager.merge_variables(
            [(g, self.component_c_variable) for g in grid_list]
        )
        self.permeability_argument = self._ad.component_c

    def _permeability(self, g: pp.Grid):
        phi = self._porosity(g)
        return self._kozeny_carman(phi)

    def _porosity(self, g):
        params = self.gb.node_props(g)[pp.PARAMETERS][self.c_parameter_key]
        rho_c = params["density_inv"]
        phi_0 = params["reference_porosity"]

        # Is this the wanted behaviour of assemble_variable, EK?
        concentration_c = self.dof_manager.assemble_variable(
            [g], self.component_c_variable, from_iterate=True
        )[self.dof_manager.grid_and_variable_to_dofs(g,
                                                     self.component_c_variable)]
        # TODO: fix ad division
        phi = phi_0 - concentration_c * rho_c
        return phi

    def _permeability_function_ad(self, concentration_c: pp.ad.Ad_array):
        subdomains = [g for g, _ in self.gb]

        rho_c = pp.ad.ParameterMatrix(self.c_parameter_key,
                                      "density_inv",
                                      subdomains,
                                      name="rho_0",
                                      ).evaluate(self.dof_manager)
        phi_0 = pp.ad.ParameterArray(self.c_parameter_key,
                                     "reference_porosity",
                                     subdomains,
                                     name="phi_0",
                                     ).parse(self.gb)
        phi_0 = pp.ad.Ad_array(phi_0, sps.csr_matrix(concentration_c.jac.shape))
        phi = self._porosity_ad(concentration_c, rho_c, phi_0)
        permeability = self._kozeny_carman(phi)
        return permeability

    def _porosity_ad(self, concentration_c, rho_c, phi_0):
        return phi_0 - rho_c * concentration_c

    def _advective_flux(self, subdomains: List[pp.Grid], component: str) -> pp.ad.Operator:
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
        setattr(self._ad, "dummy_upwind_for_discretization_" + component, upwind.upwind * var)
        flux = self._flux(subdomains)
        advection = (
                flux * (upwind.upwind * var)
                - upwind.bound_transport_dir * (flux * bc_val)
                - upwind.bound_transport_neu * bc_val
        )
        return advection

    def _assign_equations(self) -> None:
        """Define equations through discretizations.

        """
        NonlinearIncompressibleFlow._assign_equations(self)

        self._ad.time_step = pp.ad.Scalar(self.time_step, "time_step")
        transport_equation_a: pp.ad.Operator = self._subdomain_transport_equation(self.grid_list,
                                                                                  component="a")
        mass_equation_c: pp.ad.Operator = self._precipitate_mass_equation(self.grid_list, dissolved_components=["a"])
        # Assign equations to manager
        self._eq_manager.name_and_assign_equations(
            {
                "subdomain_transport_a": transport_equation_a,
                "subdomain_mass_c": mass_equation_c,
            },
        )

    def _precipitate_mass_equation(self, subdomains: List[pp.Grid], dissolved_components) -> pp.ad.Operator:
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
        rho_c = pp.ad.ParameterMatrix(self.c_parameter_key,
                                      "density_inv",
                                      subdomains,
                                      name="rho_0",
                                      )
        phi_0 = pp.ad.ParameterArray(self.c_parameter_key,
                                     "reference_porosity",
                                     subdomains,
                                     name="phi_0",
                                     )

        # Mass matrix to integrate over cell
        c_increment = mass_discr.mass * (var_c - var_c.previous_timestep())
        accumulation = (self._porosity_ad(var_c, rho_c, phi_0)
                        * c_increment
                        ) / self._ad.time_step

        reaction_term = mass_discr.mass * self._reaction_term(subdomains, dissolved_components)
        eq = accumulation - reaction_term
        return eq

    def _reaction_term(self, subdomains: List[pp.Grid], dissolved_components: List[str]) -> pp.ad.Operator:
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
        rho_c = pp.ad.ParameterMatrix(self.c_parameter_key,
                                      "density_inv",
                                      subdomains,
                                      name="rho_0",
                                      )
        phi_0 = pp.ad.ParameterArray(self.c_parameter_key,
                                     "reference_porosity",
                                     subdomains,
                                     name="phi_0",
                                     )
        r_0 = pp.ad.ParameterMatrix(self.c_parameter_key,
                                    "reference_reaction_rate",
                                    subdomains,
                                    name="r_0",
                                    )
        equilibrium_constant_inv = pp.ad.ParameterMatrix(
            param_keyword=self.c_parameter_key,
            array_keyword="equilibrium_constant_inv",
            grids=subdomains,
        )
        product = 1
        for component in dissolved_components:
            var = getattr(self._ad, "component_" + component)
            product *= var
        # product *= (1-self._ad.component_c)
        phi = self._porosity_ad(self._ad.component_c, rho_c, phi_0)
        rate = self._area_factor(phi, subdomains) * (r_0 * (equilibrium_constant_inv * product - 1))
        return rate

    def _area_factor(self, phi, subdomains):
        phi_0 = pp.ad.ParameterArray(self.c_parameter_key,
                                     "reference_porosity",
                                     subdomains,
                                     name="phi_0",
                                     )
        return phi_0 + 0.1 * (phi - phi_0)

    def _subdomain_transport_equation(self, subdomains: List[pp.Grid], component: str) -> pp.ad.Operator:
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
            grids=subdomains,
        )

        rho_c = pp.ad.ParameterMatrix(self.c_parameter_key,
                                      "density_inv",
                                      subdomains,
                                      name="rho_0",
                                      )
        phi_0 = pp.ad.ParameterArray(self.c_parameter_key,
                                     "reference_porosity",
                                     subdomains,
                                     name="phi_0",
                                     )

        div = pp.ad.Divergence(grids=subdomains)

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
        self._ad.dummy_upwind_for_discretization_a.discretize(self.gb)

    def compute_fluxes(self) -> None:
        pp.fvutils.compute_darcy_flux(
            self.gb,
            keyword=self.parameter_key,
            keyword_store=self.a_parameter_key,
            d_name="darcy_flux",
            p_name=self.variable,
            from_iterate=True,
        )

    def _export_names(self):
        return [self.variable, self.component_a_variable, self.component_c_variable, "permeability"]


class EquilibriumChemistry(Chemistry):
    def __init__(self, params: Dict):
        super().__init__(params)
        self.component_b_variable = "B"
        self.b_parameter_key = "chemistry_parameters_b"
        # Stoichiometric constant
        self._beta = self.params.get("beta", 1)
        self._equilibrium_constant = self.params.get("equilibrium_constant", .25)

    def _initial_condition(self):
        """Initial condition for all primary variables and the parameter representing the
         secondary flux variable (needs to be present for first call to upwind discretization).
        """
        # Pressure, a, b and flux for a-parameter
        super()._initial_condition()
        # add b and flux for b-parameter
        for g, d in self.gb:
            state = {self.component_b_variable: self._initial_b(g)}
            d[pp.STATE].update(state)
            d[pp.STATE][pp.ITERATE].update(state.copy())
            pp.initialize_data(g, d, self.b_parameter_key, {"darcy_flux": np.zeros(g.num_faces)})

    def _assign_variables(self) -> None:
        """
        Assign primary variables to the nodes and edges of the grid bucket.
        """
        super()._assign_variables()
        # First for the nodes
        for g, d in self.gb:
            d[pp.PRIMARY_VARIABLES].update({self.component_b_variable: {"cells": 1}})

    def _create_ad_variables(self) -> None:
        """Create the merged variables for potential and mortar flux"""
        super()._create_ad_variables()
        grid_list = [g for g, _ in self.gb.nodes()]
        self._ad.component_b = self._eq_manager.merge_variables(
            [(g, self.component_b_variable) for g in grid_list]
        )

    def _assign_equations(self) -> None:
        """Define equations through discretizations.

        Assigns a Laplace/Darcy problem discretized using Mpfa on all subdomains with
        Neumann conditions on all internal boundaries. On edges of co-dimension one,
        interface fluxes are related to higher- and lower-dimensional pressures using
        the RobinCoupling.

        Gravity is included, but may be set to 0 through assignment of the vector_source
        parameter.
        """
        NonlinearIncompressibleFlow._assign_equations(self)

        subdomain_transport_equation_a: pp.ad.Operator = self._subdomain_transport_equation(self.grid_list,
                                                                                            component="a")
        subdomain_transport_equation_b: pp.ad.Operator = self._subdomain_transport_equation(self.grid_list,
                                                                                            component="b")
        equilibrium_equation: pp.ad.Operator = self._equilibrium_equation(self.grid_list)
        # Assign equations to manager
        self._eq_manager.name_and_assign_equations(
            {
                "subdomain_transport_a": subdomain_transport_equation_a,
                "subdomain_transport_b": subdomain_transport_equation_b,
                "equilibrium": equilibrium_equation,
            },
        )

    def _equilibrium_equation(self, subdomains: List[pp.Grid]) -> pp.ad.Operator:
        """Mass balance equation for slightly compressible flow in a deformable medium.


        Parameters
        ----------
        subdomains : List[pp.Grid]
            Subdomains on which the equation is defined.

        Returns
        -------
        eq : pp.ad.Operator
            The equation on AD form.

        """
        equilibrium_constant = pp.ad.ParameterArray(
            param_keyword=self.c_parameter_key,
            array_keyword="equilibrium_constant",
            grids=subdomains,
        )
        log = pp.ad.Function(pp.ad.functions.log, "logarithm")
        eq: pp.ad.Operator = log(equilibrium_constant) - self._alpha * log(self._ad.component_a) - self._beta * log(
            self._ad.component_b)

        return eq

    def before_newton_iteration(self):
        super().before_newton_iteration()
        self._ad.dummy_upwind_for_discretization_b.discretize(self.gb)

    def compute_fluxes(self) -> None:
        super().compute_fluxes()
        # Copy to b parameters
        for _, d in self.gb:
            d[pp.PARAMETERS][self.b_parameter_key]["darcy_flux"] = d[pp.PARAMETERS][self.a_parameter_key]["darcy_flux"]

    def _export_names(self):
        return super()._export_names() + [self.component_b_variable]


class BiotNonlinearTpfa(CommonModel, pp.ContactMechanicsBiot):
    def __init__(self, params):
        super().__init__(params)

        self.time_index = 0
        self._nonlinear_iteration = 0

    def _set_scalar_parameters(self) -> None:
        super()._set_scalar_parameters()
        self._bc_map = {}

        for g, d in self.gb:
            specific_volume = self._specific_volume(g)

            comp = self.fluid.COMPRESSIBILITY
            porosity = self._porosity(g)
            phi_0 = self.rock.POROSITY * np.ones(g.num_cells)
            alpha = self._biot_alpha(g)
            bulk = pp.params.rock.bulk_from_lame(self.rock.LAMBDA, self.rock.MU)
            new_params = {}
            if g.dim == self._Nd:
                mass_weight = (
                        (porosity * comp + (alpha - phi_0) / bulk) * self.scalar_scale
                )
                new_params.update({"N_inverse": (self._biot_alpha(g) - phi_0) / bulk,
                                   "phi_reference": phi_0})
            else:
                mass_weight = porosity * comp * self.scalar_scale
            new_params.update({"mass_weight": mass_weight * specific_volume})
            parameters = d[pp.PARAMETERS][self.scalar_parameter_key]
            parameters.update(new_params)
            self._bc_map[g] = {"type": parameters["bc"],
                               "values": parameters["bc_values"],
                               }

    def _stiffness_tensor(self, g: pp.Grid) -> pp.FourthOrderTensor:
        """Fourth order stress tensor.


        Parameters
        ----------
        g : pp.Grid
            Matrix grid.

        Returns
        -------
        pp.FourthOrderTensor
            Cell-wise representation of the stress tensor.

        """
        lam = self.rock.LAMBDA * np.ones(g.num_cells)
        mu = self.rock.MU * np.ones(g.num_cells)
        return pp.FourthOrderTensor(mu, lam)

    def _permeability(self, g: pp.Grid):
        if g.dim == self._Nd:
            phi = self._porosity(g)
            vals = self._kozeny_carman(phi)
        else:
            vals = self._aperture(g) ** 2
        return vals

    def _initial_pressure(self, g):
        return np.zeros(g.num_cells)

    def _porosity_ad(self, subdomain):
        # Matrix porosity
        restriction = self._ad.subdomain_projections_scalar.cell_restriction(
            subdomain
        )
        p = restriction * self._ad.pressure
        one_by_n = pp.ad.ParameterMatrix(
            param_keyword=self.scalar_parameter_key,
            array_keyword="N_inverse",
            grids=subdomain,
        )
        p_0 = pp.ad.ParameterArray(
            param_keyword=self.mechanics_parameter_key,
            array_keyword="p_reference",
            grids=subdomain,
        )
        phi_0 = pp.ad.ParameterArray(
            param_keyword=self.scalar_parameter_key,
            array_keyword="phi_reference",
            grids=subdomain,
        )
        phi = phi_0 + self._div_u(subdomain) + one_by_n * (p - p_0)
        return phi

    def _porosity(self, g):
        if g.dim < self._Nd:
            vals = np.ones(g.num_cells)
        elif self.time_index == 0 and self._nonlinear_iteration == 0:
            vals = self.rock.POROSITY * np.ones(g.num_cells)
        else:
            vals = self._porosity_ad([g]).evaluate(self.dof_manager).val
        return vals

    def _permeability_function_ad(self, var: pp.ad.Ad_array):
        phi = self._porosity_ad([self._nd_grid()]).evaluate(self.dof_manager)
        fracture_subdomains: List[pp.Grid] = self.gb.grids_of_dimension(
            self._Nd - 1
        ).tolist()
        u_n = (
                self._ad.normal_component_frac
                * self._displacement_jump(fracture_subdomains)
        )
        max_ad = pp.ad.Function(pp.ad.functions.maximum, "maximum")
        u_n = max_ad(u_n, self._gap(fracture_subdomains)).evaluate(self.dof_manager)
        prolongation_frac = self._ad.subdomain_projections_scalar.cell_prolongation(
            fracture_subdomains
        ).evaluate(self.dof_manager)
        prolongation_matrix = self._ad.subdomain_projections_scalar.cell_prolongation(
            [self._nd_grid()]
        ).evaluate(self.dof_manager)
        frac = u_n * u_n * u_n
        val = prolongation_frac * frac + prolongation_matrix * self._kozeny_carman(phi)
        return val

    def _fluid_flux(self, subdomains: List[pp.Grid]) -> pp.ad.Operator:

        if self.params.get("use_tpfa", False) or self.gb.dim_max() == 1:
            base_discr = pp.ad.TpfaAd(self.scalar_parameter_key, subdomains)
        else:
            base_discr = pp.ad.MpfaAd(self.scalar_parameter_key, subdomains)
        self._ad.flux_discretization = base_discr

        TPFA_function = pp.ad.DifferentiableFVAd(
            grid_list=subdomains,
            bc=self._bc_map,
            base_discr=base_discr,
            dof_manager=self.dof_manager,
            var_name=self.scalar_variable,
        )
        p = self._ad.pressure
        u_j = self._ad.interface_displacement
        perm_function = pp.ad.Function(
            self._permeability_function_ad, "perm_function_biot"
        )
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

    def _aperture(self, g: pp.Grid) -> np.ndarray:
        """
        Aperture is a characteristic thickness of a cell, with units [m].
        1 in matrix, thickness of fractures and "side length" of cross-sectional
        area/volume (or "specific volume") for intersections of co-dimension 2 and 3.
        See also specific_volume.
        """
        aperture = np.ones(g.num_cells)
        if g.dim < self._Nd:
            data_edge = self.gb.edge_props((g, self._nd_grid()))
            proj = self.gb.node_props(g)["tangential_normal_projection"]
            u_j = self.reconstruct_local_displacement_jump(data_edge, proj)
            aperture = np.max(np.vstack((u_j[-1], self._initial_gap(g))), axis=0)
        return aperture

    def _initial_condition(self):
        """Initial condition for all primary variables and the parameter representing the
         secondary flux variable (needs to be present for first call to upwind discretization).
        """
        super()._initial_condition()
        for g, d in self.gb:
            state = {
                self.scalar_variable: self._initial_pressure(g),
            }
            d[pp.STATE].update(state)
            d[pp.STATE][pp.ITERATE].update(state.copy())

    def after_newton_iteration(self, solution_vector: np.ndarray) -> None:
        """Prepare for plotting of displacements.

        This is done after each iteration in order to facilitate export of
        non-converged solutions.

        Parameters:
            solution_vector (np.array): solution vector for the current iterate.

        """
        super().after_newton_iteration(solution_vector)
        nd = self._Nd
        for g, d in self.gb:
            displacements = np.zeros((3, g.num_cells))
            if g.dim == nd:
                displacements[:nd] = d[pp.STATE][pp.ITERATE][self.displacement_variable].reshape((nd, g.num_cells),
                                                                                                 order="F")

            d[pp.STATE]["displacements"] = displacements
            d[pp.STATE]["aperture"] = self._aperture(g)

    def _export_names(self):
        return [self.scalar_variable, "displacements", "aperture", "permeability"]
