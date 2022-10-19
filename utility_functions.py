import pickle
from typing import Dict, Tuple

import matplotlib
import numpy as np
import porepy as pp
from matplotlib import pyplot as plt

matplotlib.use("Qt5Agg")
plt.rc("font", size=16)
legend_fontsize = 14


def read_pickle(fn: str):
    f = open(fn, "rb")
    content = pickle.load(f)
    f.close()
    return content


def write_pickle(content, fn: str):
    f = open(fn, "wb")
    pickle.dump(content, f)
    f.close()


def store_converged_permeability(model):
    perm_dict = dict()
    for sd, data in model.mdg.subdomains(return_data=True):
        perm_dict[sd.dim] = data[pp.PARAMETERS]["flow"]["second_order_tensor"].values[
            0, 0
        ]
    write_pickle(perm_dict, "converged_permeabilities/" + model.params["file_name"])


def load_converged_permeability(model):
    fn = "converged_permeabilities/" + model.params["file_name"]
    try:
        perm_dict = read_pickle(fn)
    except:
        perm_dict = {}
        for sd in model.mdg.subdomains():
            perm_dict[sd.dim] = np.inf

    model.converged_permeabilities = perm_dict
    for sd, data in model.mdg.subdomains(return_data=True):
        data[pp.PARAMETERS]["flow"]["converged_permeability"] = perm_dict[sd.dim]


def plot_permeability_errors(
    m_nonlin, m_lin, color=None, nd=True, use_fn_label=True, model_type="both"
):
    if nd:
        vals_lin = m_lin._permeability_errors_nd[:-1]
        vals_nonlin = m_nonlin._permeability_errors_nd[:-1]
    else:
        vals_lin = m_lin._permeability_errors_frac[:-1]
        vals_nonlin = m_nonlin._permeability_errors_frac[:-1]
    iterations = np.arange(1, 1 + len(vals_lin))
    iterations_nonl = np.arange(1, 1 + len(vals_nonlin))
    if use_fn_label:
        label = m_lin.params["file_name"]
    else:
        label = "diff"
    if model_type != "linear":
        plt.semilogy(
            iterations_nonl,
            vals_nonlin,
            label=label,
            ls="--",
            color=color,
            linewidth=2,
        )
    if not use_fn_label:
        label = "non-diff"
    if model_type != "nonlinear":
        plt.semilogy(
            iterations, vals_lin, label=label, ls="-", color=color, linewidth=2
        )
    ax = plt.gca()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Permeability error norm")
    # Indicate convergence order
    plt.legend(
        title=m_lin.params.get("legend_title", None),
        loc="upper right",
        fontsize=legend_fontsize,
        title_fontsize=legend_fontsize,
    )
    plt.tight_layout()
    fn_root = "images/permeability_errors_"
    if not nd:
        fn_root += "frac_"
    plt.savefig(fname=fn_root + m_nonlin.params["plotting_file_name"])


def plot_convergence(
    m_nonlin, m_lin, color=None, plot_errors=False, use_fn_label=True, model_type="both"
):
    iterations = np.arange(1, len(m_lin._residuals) + 1)
    iterations_nonl = np.arange(1, len(m_nonlin._residuals) + 1)
    if "legend_label" in m_lin.params:
        label = m_lin.params["legend_label"] + "D"
    elif use_fn_label:
        label = m_lin.params["file_name"]
    else:
        label = "diff"

    if "Tetra" in m_lin.params["file_name"]:
        marker = "v"
    else:
        marker = None

    if model_type != "linear":
        plt.semilogy(
            iterations_nonl,
            m_nonlin._residuals,
            label=label,
            ls="--",
            color=color,
            linewidth=2,
            marker=marker,
            markersize=7,
        )
    if not use_fn_label:
        label = "non-diff"
    if model_type != "nonlinear":
        if "legend_label" in m_lin.params:
            label = label[:-1]
        plt.semilogy(
            iterations,
            m_lin._residuals,
            label=label,
            ls="-",
            color=color,
            linewidth=2,
            marker=marker,
            markersize=7,
        )
    ax = plt.gca()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual norm")
    # Indicate convergence order
    plt.legend(
        title=m_lin.params.get("legend_title", None),
        loc="upper right",
        fontsize=legend_fontsize,
        title_fontsize=legend_fontsize,
    )
    plt.tight_layout()
    plt.savefig(fname="images/residuals_" + m_nonlin.params["plotting_file_name"])


# p_analytical = m_nonlin.p_analytical()
#         e_nonlin = [
#             np.linalg.norm(p_num - p_analytical) for p_num in m_nonlin._solutions
#         ]
#         e_lin = [np.linalg.norm(p_num - p_analytical) for p_num in m_lin._solutions]
#
#         plt.semilogy(iterations, e_lin, label="linear")
#         plt.semilogy(iterations_nonl, e_nonlin, label="nonlinear")
#         ax = plt.gca()
#         ax.set_xlabel("Iteration")
#         ax.set_ylabel("Error (not normalized)")
#         plt.legend(fontsize=legend_fontsize)


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


def plot_all_permeability_errors(update_params):
    line_colors = ["blue", "orange", "green", "red", "purple", "brown"]
    i = 0
    plt.figure()
    for key in update_params.keys():
        m_lin, m_nonlin = update_params[key]["models"]
        plot_permeability_errors(m_nonlin, m_lin, color="tab:" + line_colors[i])
        i += 1


def plot_multiple_time_steps(
    updates: dict, n_steps: int, use_fn_label: bool = True, model_type: str = "both"
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
            if hasattr(nonlinear_model, "residual_list"):
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

    return linear_model, nonlinear_model


def extract_line_solutions(model):
    if not hasattr(model, "iteration_permeability"):
        model.iteration_permeability = []
        model.iteration_pressure = []
        model.iteration_c = []
    for sd, data in model.mdg.subdomains(return_data=True):
        cell_inds = np.logical_and(
            np.isclose(sd.cell_centers[2], 0.5), np.isclose(sd.cell_centers[1], 0.5)
        )
        if np.any(cell_inds):
            model.iteration_permeability.append(model._permeability(sd)[cell_inds])
            if hasattr(model, "variable"):
                model.iteration_pressure.append(
                    data[pp.STATE][pp.ITERATE][model.variable][cell_inds]
                )
            else:
                model.iteration_pressure.append(
                    data[pp.STATE][pp.ITERATE][model.scalar_variable][cell_inds]
                )
            if hasattr(model, "variable"):
                model.iteration_c.append(
                    data[pp.STATE][pp.ITERATE][model.component_c_variable][cell_inds]
                )


def plot_line_solutions(m_nonlin, m_lin, color=None, model_type="both", fname=None):
    from_iteration = 0
    vals_nonl = m_nonlin.iteration_pressure[from_iteration:]
    vals_lin = m_lin.iteration_pressure[from_iteration:]
    difference = True
    for sd, data in m_lin.mdg.subdomains(return_data=True):
        cell_inds = np.logical_and(
            np.isclose(sd.cell_centers[2], 0.5), np.isclose(sd.cell_centers[1], 0.5)
        )
        x = sd.cell_centers[0, cell_inds]

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
            plt.legend(
                title=m_lin.params.get("legend_title", None),
                loc="upper right",
                fontsize=legend_fontsize,
            )
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
            plt.legend(
                title=m_lin.params.get("legend_title", None),
                loc="upper right",
                fontsize=legend_fontsize,
            )
            # if fname is None:
            fname = (
                "images/iterations/permeabilities_linear"
                + m_nonlin.params["plotting_file_name"]
            )
            plt.savefig(fname=fname + str(i))


# Profiling: -B -m cProfile -o OutputFileName
