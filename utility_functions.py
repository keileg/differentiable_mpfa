import copy
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
    m_nonlin, m_lin, color=None, use_fn_label=True, model_type="both"
):
    vals_lin = m_lin._permeabilities_nd[:-1]
    k_converged = m_nonlin._permeabilities_nd[-1]
    diffs_lin = k_converged - np.vstack(vals_lin)
    perm_errors_lin = np.linalg.norm(diffs_lin, axis=1) / np.sqrt(k_converged.size)
    vals_nonl = m_nonlin._permeabilities_nd[:-1]
    diffs_nonl = k_converged - np.vstack(vals_nonl)
    perm_errors_nonl = np.linalg.norm(diffs_nonl, axis=1) / np.sqrt(k_converged.size)
    iterations = np.arange(1, 1 + len(vals_lin))
    iterations_nonl = np.arange(1, 1 + len(vals_nonl))
    if use_fn_label:
        label = m_lin.params["file_name"]
    else:
        label = "diff"
    if model_type != "linear":
        plt.semilogy(
            iterations_nonl,
            perm_errors_nonl,
            label=label,
            ls="--",
            color=color,
            linewidth=2,
        )
    if not use_fn_label:
        label = "non-diff"
    if model_type != "nonlinear":
        plt.semilogy(
            iterations, perm_errors_lin, label=label, ls="-", color=color, linewidth=2
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


def domain_bounds_to_array(domain_bounds) -> np.ndarray:
    """Obtain points defining the domain_bounds as a single array.

    Parameters:
        domain_bounds: Dictionary containing keys xmin and xmax and optionally
            ymin, ymax, zmin, zmax.

    Returns:
        Bounds as an nd x 2 array. nd should be between 1 and 3.

    """
    pts = np.atleast_2d([[domain_bounds["xmin"], domain_bounds["xmax"]]])
    if "ymin" in domain_bounds:
        pts = np.vstack(
            (pts, np.atleast_2d([[domain_bounds["ymin"], domain_bounds["ymax"]]]))
        )
    if "zmin" in domain_bounds:
        pts = np.vstack(
            (pts, np.atleast_2d([[domain_bounds["zmin"], domain_bounds["zmax"]]]))
        )
    return pts


def run_simulation_pairs_varying_parameters(
    params: Dict, update_params: Dict[str, Dict], model_class
) -> None:
    """Run multiple pairs of simulations varying some parameters.

    Parameters:
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
        params_loc = copy.deepcopy(params)
        params_loc.update(updates)
        params_loc["file_name"] = name
        print("Running simulation named", name)
        update_params[name]["models"] = run_linear_and_nonlinear(
            params_loc, model_class, "tab:" + line_colors[i]
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


def run_linear_and_nonlinear(
    params: Dict, model_class, plot_color=None, use_fn_label=True
) -> Tuple:
    """Run a pair of one linear and one nonlinear simulation.

    Parameters:
        params: Setup and run parameters
        model_class: Model to be used for the simulations
        plot_color: Color for residual plots
        use_fn_label: Label the plots using params["file_name"]

    Returns:
        nonlinear_model, linear_model

    """
    params_linear = copy.deepcopy(params)
    params_linear["use_linear_discretization"] = True
    linear_model = model_class(params_linear)
    run_method = pp.run_time_dependent_model
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
