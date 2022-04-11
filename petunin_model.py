import porepy as pp
from grids import one_dimensional_grid_bucket
from incompressible_flow_nonlinear_tpfa import NonlinearIncompressibleFlow
from plotting import plot_convergence

class PetuninPermeability:
    def _permeability_function(self, pressure: pp.ad.Variable):

        phi_0 = .1
        pressure_0 = pp.BAR
        c_r = 10 * pp.PASCAL
        kappa_0 = pp.MILLIDARCY
        alpha = 2
        kappa = kappa_0 * (1 + c_r * (pressure - pressure_0) / phi_0) ** alpha
        return kappa


if __name__ == "__main__":
    num_iterations = 15
    params = {
        "use_tpfa": True,
        "linear_solver": "pypardiso",
        "max_iterations": num_iterations,
        "nl_convergence_tol": 1e-10,
        "nl_divergence_tol": 1e5,
        "grid_method": one_dimensional_grid_bucket,
    }
    nonlinear_model = NonlinearIncompressibleFlow(params)
    pp.run_stationary_model(nonlinear_model, params)
    params_linear = params.copy()
    params_linear["use_linear_discretization"] = True
    linear_model = NonlinearIncompressibleFlow(params_linear)
    pp.run_stationary_model(linear_model, params_linear)
    plot_convergence(nonlinear_model, linear_model, plot_errors=False)
