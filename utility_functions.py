from typing import Dict, Tuple

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

# from mpltools import annotation
import porepy as pp

matplotlib.use("Qt5Agg")


def plot_convergence(
    m_nonlin, m_lin, color=None, plot_errors=False, use_fn_label=True, model_type="both"
):
    iterations = np.arange(len(m_lin._residuals))
    iterations_nonl = np.arange(len(m_nonlin._residuals))
    if use_fn_label:
        label = m_lin.params["file_name"]
    else:
        label = "diff"
    if model_type != "linear":
        plt.semilogy(
            iterations_nonl,
            m_nonlin._residuals,
            label=label,
            ls="--",
            color=color,
            marker="x",
        )
    if not use_fn_label:
        label = "non-diff"
    if model_type != "nonlinear":
        plt.semilogy(
            iterations, m_lin._residuals, label=label, ls="-", color=color, marker="+"
        )
    ax = plt.gca()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual norm")
    # Indicate convergence order
    plt.legend(title=m_lin.params.get("legend_title", None), loc="upper right")
    plt.savefig(fname="images/residuals_" + m_nonlin.params["plotting_file_name"])
    if plot_errors:
        plt.figure()
        p_analytical = m_nonlin.p_analytical()
        e_nonlin = [
            np.linalg.norm(p_num - p_analytical) for p_num in m_nonlin._solutions
        ]
        e_lin = [np.linalg.norm(p_num - p_analytical) for p_num in m_lin._solutions]

        plt.semilogy(iterations, e_lin, label="linear")
        plt.semilogy(iterations_nonl, e_nonlin, label="nonlinear")
        ax = plt.gca()
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Error (not normalized)")
        plt.legend()


def run_simulation_pairs_varying_parameters(
    params: Dict, update_params: Dict[str, Dict], model_class
) -> None:
    """Run multiple pairs of simulations varying some parameters.

    Args:
        params: Dictionary containing initial setup and run parameters
        update_params: Dictionary with keys identifying each simulation pair and
            values being dictionaries specifying the updates to the initial parameters.

    Example usage:
    The following parameters produce four simulations (linear + 100 cells,
    nonlinear + 100 cells, linear + 4 cells and nonlinear + 4 cells):

        params = {"foo": "bar", "grid_method": two_dimensional_cartesian, ...}
        update_params = {"hundred_cells": {"n_cells": [10, 10]},
                         "four_cells": {"n_cells": [2, 2]},
                         }
    """
    line_colors = ["blue", "orange", "green", "red", "purple", "brown"]
    i = 0
    plt.figure()
    for name, updates in update_params.items():
        params.update(updates)
        params["file_name"] = name
        print("Running simulation named", name)
        update_params[name]["models"] = run_linear_and_nonlinear(
            params, model_class, "tab:" + line_colors[i]
        )
        i += 1


def plot_multiple_time_steps(
    updates: Dict, n_steps: int, use_fn_label: bool = True, model_type: str = "both"
):
    """

    Args:
        updates: Dictionary with keys identifying the different simulations. For each key, updates[key]["models"]
            contains a tuple of model instances resulting from simulations with and without full flux differentiation.
        n_steps: Number of time steps
        use_fn_label: Whether to use the file name as a label in the plots.
        model_type: Which of the two model variants to plot results for. If specified as either "linear"
            or "nonlinear", the other one is omitted.

    Returns:

    """

    line_colors = ["blue", "orange", "green", "red", "purple", "brown"]

    for time_step in range(n_steps):
        plt.figure()
        for i, name in enumerate(updates.keys()):
            linear_model, nonlinear_model = updates[name]["models"]
            nonlinear_model._residuals = nonlinear_model.residual_list[time_step]
            linear_model._residuals = linear_model.residual_list[time_step]
            nonlinear_model.params["plotting_file_name"] = (
                linear_model.params["plotting_file_name"] + "_" + str(time_step)
            )
            plot_convergence(
                nonlinear_model,
                linear_model,
                plot_errors=False,
                color="tab:" + line_colors[i],
                use_fn_label=use_fn_label,
                model_type=model_type,
            )


def run_linear_and_nonlinear(
    params: Dict, model_class, plot_color=None, use_fn_label=True
) -> Tuple:
    """Run a pair of one linear and one nonlinear simulation.

    Args:
        params: Setup and run parameters
        model_class: Model to be used for the simulations
        plot_color: Color for residual plots
        use_fn_label: Label the plots using params["file_name"]

    Returns:
        nonlinear_model, linear_model

    """
    params_linear = params.copy()
    params_linear["use_linear_discretization"] = True
    linear_model = model_class(params_linear)
    if "time_step" in linear_model.params:
        run_method = pp.run_time_dependent_model
    else:
        run_method = pp.run_stationary_model
    params["file_name"] += "_nonlinear"
    nonlinear_model = model_class(params)
    run_method(nonlinear_model, params)
    run_method(linear_model, params_linear)
    plot_convergence(
        nonlinear_model,
        linear_model,
        plot_errors=False,
        color=plot_color,
        use_fn_label=use_fn_label,
    )
    # for m in ["linear", "nonlinear"]:
    #     fname = "images/iterations/permeabilities_" + nonlinear_model.params["plotting_file_name"] + m
    #     plot_line_solutions(nonlinear_model, linear_model, model_type=m, fname=fname)
    return linear_model, nonlinear_model


def extract_line_solutions(model):
    if not hasattr(model, "iteration_permeability"):
        model.iteration_permeability = []
        model.iteration_pressure = []
        model.iteration_c = []
    for g, d in model.gb:
        cell_inds = np.logical_and(
            np.isclose(g.cell_centers[2], 0.5), np.isclose(g.cell_centers[1], 0.5)
        )
        if np.any(cell_inds):
            model.iteration_permeability.append(model._permeability(g)[cell_inds])
            if hasattr(model, "variable"):
                model.iteration_pressure.append(
                    d[pp.STATE][pp.ITERATE][model.variable][cell_inds]
                )
            else:
                model.iteration_pressure.append(
                    d[pp.STATE][pp.ITERATE][model.scalar_variable][cell_inds]
                )
            if hasattr(model, "variable"):
                model.iteration_c.append(
                    d[pp.STATE][pp.ITERATE][model.component_c_variable][cell_inds]
                )


def plot_line_solutions(m_nonlin, m_lin, color=None, model_type="both", fname=None):
    # if use_fn_label:
    #     label = m_lin.params["file_name"]
    # else:
    from_iteration = 0
    vals_nonl = m_nonlin.iteration_pressure[from_iteration:]
    vals_lin = m_lin.iteration_pressure[from_iteration:]
    difference = True
    for g, d in m_lin.gb:
        cell_inds = np.logical_and(
            np.isclose(g.cell_centers[2], 0.5), np.isclose(g.cell_centers[1], 0.5)
        )
        x = g.cell_centers[0, cell_inds]

    c_min = np.inf
    c_max = -np.inf
    for i, perms in enumerate(vals_lin):
        if difference:
            c_min = min(c_min, np.min(perms - vals_lin[-1]))
            c_max = max(c_max, np.max(perms - vals_lin[-1]))
    if model_type != "linear":
        for i, perms in enumerate(vals_nonl):
            plt.figure()
            if difference:
                plt.plot(
                    x,
                    perms - vals_nonl[-1],
                    label="Iteration " + str(i),
                    ls="--",
                    color=color,
                    marker="x",
                )
            else:
                plt.plot(
                    x,
                    perms,
                    label="Iteration " + str(i),
                    ls="--",
                    color=color,
                    marker="x",
                )
                plt.plot(
                    x, vals_nonl[-1], label="Converged", ls=":", color=color, marker="x"
                )

            ax = plt.gca()
            ax.set_xlabel("x coordinate")
            ax.set_ylabel("K")
            plt.ylim(c_min, c_max)
            # Indicate convergence order
            plt.legend(title=m_lin.params.get("legend_title", None), loc="upper right")
            # if fname is None:
            fname = (
                "images/iterations/permeabilities_nonlinear"
                + m_nonlin.params["plotting_file_name"]
            )
            plt.savefig(fname=fname + str(i))
    # if not use_fn_label:
    if model_type != "nonlinear":

        for i, perms in enumerate(vals_lin):
            plt.figure()
            if difference:
                plt.plot(
                    x,
                    perms - vals_lin[-1],
                    label="Iteration " + str(i),
                    ls="--",
                    color=color,
                    marker="x",
                )
            else:
                plt.plot(
                    x,
                    perms,
                    label="Iteration " + str(i),
                    ls="-",
                    color=color,
                    marker="+",
                )
                plt.plot(
                    x, vals_lin[-1], label="Converged", ls=":", color=color, marker="x"
                )
            ax = plt.gca()
            ax.set_xlabel("x coordinate")
            ax.set_ylabel("K")
            plt.ylim(c_min, c_max)

            # Indicate convergence order
            plt.legend(title=m_lin.params.get("legend_title", None), loc="upper right")
            # if fname is None:
            fname = (
                "images/iterations/permeabilities_linear"
                + m_nonlin.params["plotting_file_name"]
            )
            plt.savefig(fname=fname + str(i))


# Profiling: -B -m cProfile -o OutputFileName
