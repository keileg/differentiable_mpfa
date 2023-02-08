"""
Models using differentiable Tpfa/Mpfa.

Four specific simulation models are constructed:
    Mass balance for incompressible fluid.
    Reactive transport.
    Monodimensional poroelasticity.
    Poroelasticity with fracture deformation.
"""
import logging
from typing import Any, Callable, Tuple

import numpy as np
import porepy as pp

from constitutive_laws import (
    DifferentiatedDarcyLaw,
    DomainCenterSource,
    PoromechanicsPermeability,
)
from grids import Geometry
from utility_functions import load_converged_permeability

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


solid_values = {
    "permeability": 1e-4,
    "density": 1,
    "porosity": 0.05,
    "shear_modulus": 1e3,
    "lame_lambda": 1e3,
}


class DataSaving(pp.viz.data_saving_model_mixin.DataSavingMixin):
    """Mixin class for data saving."""

    permeability: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Permeability."""
    porosity: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Porosity."""
    aperture: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Aperture."""
    nd: int
    """Number of dimensions."""
    _residual: Callable[[], float]
    """Method returning current residual."""
    equation_system: pp.EquationSystem
    """Equation system manager."""
    darcy_keyword: str
    """Discretization keyword for Darcy flux."""

    def initialize_data_saving(self) -> None:
        """Initialize data saving.

        This method is called by :meth:`prepare_simulation` to initialize the exporter,
        and any other data saving functionality (e.g., empty data containers to be
        appended in :meth:`save_data_time_step`).
        """
        super().initialize_data_saving()
        self._solutions: list[np.ndarray] = []
        self._residuals: list[float] = []
        self._permeability_errors_nd: list[float] = []
        self._permeability_errors_frac: list[float] = []
        # Add the permeability and porosity to the data to be exported.
        self.equation_system.discretize()
        for sd, data in self.mdg.subdomains(return_data=True):
            self._store_secondary_values_in_state(sd, data)

    def _store_secondary_values_in_state(self, sd: pp.Grid, data: dict) -> None:
        """Store secondary variables in the state.

        This method is called by :meth:`save_data_time_step` to store the current
        values of the secondary variables in the state dictionary.
        """
        method_names = ["permeability", "porosity", "aperture"]
        # Discretize to ensure that matrices needed for e.g. porosity
        # evaluation are available.
        for method_name in method_names:
            if hasattr(self, method_name):
                method = getattr(self, method_name)
                value = method([sd]).evaluate(self.equation_system)
                if isinstance(value, pp.ad.Ad_array):
                    value = value.val
                # Broadcast scalars to cell-wise values.
                if isinstance(value, (float, int)):
                    value = np.ones(sd.num_cells) * value
                data[pp.STATE][method_name] = value

    def save_data_iteration(self) -> None:
        """Export the model state at a given time step."""
        for sd, data in self.mdg.subdomains(return_data=True):
            self._store_secondary_values_in_state(sd, data)
        self._residuals.append(self._residual())
        sol = self.equation_system.get_variable_values()
        self._solutions.append(sol)

    def data_to_export(self):
        """Return data to be exported.

        Return type should comply with pp.exporter.DataInput.

        Returns:
            List containing names of the quantities to be exported.

        """
        # Primary variables.
        var_names = super().data_to_export()
        # Add the permeability and porosity.
        names = ["permeability", "porosity"] + var_names
        if hasattr(self, "aperture"):
            names.append("aperture")
        return names


class SolutionStrategyMixin:
    mdg: pp.MixedDimensionalGrid
    nd: int
    domain_bounds: dict
    equation_system: pp.EquationSystem
    darcy_keyword: str
    bc_values_darcy: Callable[[list[pp.Grid]], pp.ad.Operator]
    params: dict[str, Any]
    pressure: Callable[[list[pp.Grid]], pp.ad.Operator]
    save_data_iteration: Callable[[], None]
    save_data_time_step: Callable[[], None]

    def __init__(self, params: dict):
        # Common solver parameters:
        default_params = {
            "linear_solver": "pypardiso",
            "max_iterations": 24,
            "nl_divergence_tol": 1e5,
            "nl_convergence_tol": 1e-12,
        }
        for key, val in default_params.items():
            if key not in params:
                params[key] = val
        super().__init__(params)  # type: ignore

    def set_discretization_parameters(self) -> None:
        """Set default (unitary/zero) parameters for the flow problem.

        In addition to whatever super might do, set boundary conditions for the
        Darcy flux used by the differentiated tpfa discretization.

        """
        super().set_discretization_parameters()
        for sd, data in self.mdg.subdomains(return_data=True):
            vals = self.bc_values_darcy([sd])
            pp.initialize_data(
                sd,
                data,
                self.darcy_keyword,
                {
                    "bc_values": vals.evaluate(self.equation_system),
                },
            )

    def prepare_simulation(self):
        super().prepare_simulation()
        if self.params.get("compute_permeability_errors", False):
            load_converged_permeability(self)

    def before_nonlinear_iteration(self) -> None:
        """Set parameters and rediscretize."""
        self.set_discretization_parameters()
        super().before_nonlinear_iteration()

    def after_nonlinear_iteration(self, solution_vector: np.ndarray) -> None:
        """Save data and check for convergence."""
        super().after_nonlinear_iteration(solution_vector)
        self.save_data_iteration()

    def check_convergence(
        self,
        solution: np.ndarray,
        prev_solution: np.ndarray,
        init_solution: np.ndarray,
        nl_params: dict[str, Any],
    ) -> Tuple[float, bool, bool]:
        """Check convergence based on current residual.

        Parameters:
            solution: Newly obtained solution vector.
            prev_solution: Solution obtained in the previous non-linear
                iteration.
            init_solution: Solution obtained from the previous time-step.
            nl_params (dict): Dictionary of parameters used for the convergence check.
                Which items are required will depend on the convergence test to be
                implemented.

        Returns:
            float: Error, computed to the norm in question.
            boolean: True if the solution is converged according to the test
                implemented by this method.
            boolean: True if the solution is diverged according to the test
                implemented by this method.

        """
        error = self._residual()
        logger.info(f"Normalized error: {error:.2e}")
        converged = error < nl_params["nl_convergence_tol"]
        diverged = False
        return error, converged, diverged

    def _residual(self):
        """Compute the residual of the linear system.

        Returns:
            The residual, normalized by the size of the solution vector.

        """
        # Update parameters and rediscretize.
        self.set_discretization_parameters()
        self.discretize()
        # Assemble the linear system and access the residual.
        self.assemble_linear_system()
        _, b = self.linear_system
        # Compute the residual, normalized by the size of the solution vector.
        normalized_residual = np.linalg.norm(b) / np.sqrt(b.size)

        return normalized_residual

    def after_nonlinear_failure(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """Called if the non-linear solver fails to converge.

        Parameters:
            solution: Solution vector at the time of failure.
            errors: Error at the time of failure.
            iteration_counter: Number of iterations at the time of failure.

        """
        tol = self.params["nl_convergence_tol"]
        logger.info(f"Newton iterations did not converge to tolerance {tol}.")
        self.save_data_time_step()

    def permeability_argument(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Permeability argument for the Darcy flux.

        TODO: Legacy reasons for including this. New version of diff-tpfa only
        uses this to infer size of Jacobian. Should be removed if possible.

        Parameters:
            subdomains: List of subdomains where the Darcy flux is defined.

        Returns:
            Permeability argument.

        """
        return self.pressure(subdomains)


class Poromechanics(
    DataSaving,
    Geometry,
    PoromechanicsPermeability,
    DifferentiatedDarcyLaw,
    SolutionStrategyMixin,
    DomainCenterSource,
    pp.poromechanics.Poromechanics,
):
    """Poromechanics model with differentiated Darcy law.

    Modification from the standard poromechanics model in porepy is in the constitutive
    law for Darcy flux and its implication on solution strategy. Also includes bespoke
    data saving and geometry methods.

    """

    pass
