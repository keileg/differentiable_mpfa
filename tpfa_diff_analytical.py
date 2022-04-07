"""
Script to develop a discretization of differentiable Tpfa/Mpfa.
"""
from typing import Union, List
from functools import partial
import porepy as pp
import numpy as np
import scipy.sparse as sps
import sympy as sym
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from pypardiso import spsolve as pardisosolve

nsteps = 5
use_ad = 1==1
g1 = pp.CartGrid([15], physdims=[1])
gb = pp.GridBucket()
gb.add_nodes([g1])
gb.assign_node_ordering()
gb.compute_geometry()

grid_list = [g for g, _ in gb]

keyword = "flow"
variable = "p"
# EK: three_vars, and the variables u1 and u2, was used previously to ensure what we do
# make sense for the multiple variables. It can probably be deleted, but also be expanded
# to test for more complex physics
linear_perm = True


if linear_perm:
    def permeability_from_pressure(var):
        return 1 + var


    def source_function(x):
        return - 1 - 6 * (x - x * x)
else:
    def permeability_from_pressure(var):
        return 1 + var + var * var


    def source_function(x):
        return - 2 + 2 * x - 12 * x * x + 20 * x * x * x - 10 * x ** 4 + ( 1 - 6 * x + 6 * x ** 2)

    # def permeability_from_pressure(var):
    #     return 1 + var *  var
    #
    #
    # def source_function(x):
    #     return - 2 + 2 * x - 12 * x * x + 20 * x * x * x - 10 * x ** 4


# def p_analytical(x):
#     return (1 - x) * x
def sqrt(var):
    return var ** (1/2)

def source_function(x):
   return -2*sqrt(x*(1 - x)) - 0.2 + sqrt(x*(1 - x))*(1/2 - x)*(1 - 2*x)/(x*(1 - x))
def p_analytical(x):
   return x*(1 - x)
def permeability_from_pressure(var):
   return sqrt(var) + 0.1

# The Tpfa discretization needs to know which faces are on the boundary for each grid.
# This means that having access to Ad Operator formulation is insufficient (this takes)
# boundary values only. As a temporary construct, we pass a dictionary from grids to
# boundary conditions.
bc_map = {}

# initialize data. This is quite random.
for g, data in gb:
    data[pp.PRIMARY_VARIABLES] = {
        variable: {"cells": 1},
    }
    x, y = g.cell_centers[0], g.cell_centers[1]

    initial_value = 10. * np.ones(g.num_cells)   # p_analytical(x)
    data[pp.STATE] = {
        variable: initial_value.copy(),
        pp.ITERATE: {
            variable: initial_value.copy(),
        },
    }

    bound_faces = g.tags["domain_boundary_faces"]
    bc = pp.BoundaryCondition(g, bound_faces, bound_faces.sum() * ["dir"])

    bc_values = np.zeros(g.num_faces)
    bc_values[bound_faces] = p_analytical(g.face_centers[0, bound_faces])
    print(bc_values)
    bc_map[g] = (bc) #, bc_values)
    permeability = pp.SecondOrderTensor(permeability_from_pressure(initial_value))
    source_values = np.zeros(g.num_cells)
    x, y = g.cell_centers[0], g.cell_centers[1]
    source_values = -source_function(x) * g.cell_volumes
    specified_data = {
        "bc": bc,
        "bc_values": bc_values,
        "second_order_tensor": permeability,
        "source": source_values,
    }

    pp.initialize_data(g, data, keyword, specified_parameters=specified_data)

def setup_diff_problem(gb, fv_scheme="tp"):
    dof_manager = pp.DofManager(gb)
    eq_manager = pp.ad.EquationManager(gb, dof_manager)

    p = eq_manager.merge_variables([(g, variable) for g in grid_list])
    if fv_scheme=="tp":
        base_discr = pp.ad.TpfaAd(keyword, grid_list)
    else:
        base_discr = pp.ad.MpfaAd(keyword, grid_list)

    div = pp.ad.Divergence(grid_list)

    proj = pp.ad.SubdomainProjections(grid_list)

    TPFA = pp.ad.Function(
        partial(
            # For the moment, the actual implementation is a function differentiable_mpfa
            # located in pp.ad.discretizations. Both should be revised.
            pp.ad.differentiable_mpfa,
            grid_list=grid_list,
            bc=bc_map,
            base_discr=base_discr,
            dof_manager=dof_manager,
            var_name=variable,
            projections=proj,
        ),
        "tpfa_ad",
    )
    flux_ad = TPFA(pp.ad.Function(permeability_from_pressure, "perm_function"), p, p)

    bc = pp.ad.ParameterArray(
        keyword,
        array_keyword="bc_values",
        grids=grid_list,
    )
    source = pp.ad.ParameterArray(
        param_keyword=keyword,
        array_keyword="source",
        grids=grid_list,
    )
    eq_ad = div * (flux_ad + base_discr.bound_flux * bc) - source
    # Make an Ad operator which contains the base discretization, so that we have something
    # to discretize (with the current implementation, the base discretization will not be
    # seen by calling eq.discretize(gb)). TODO..
    flux_tpfa = base_discr.flux * p
    eq = div * (base_discr.flux * p + base_discr.bound_flux * bc) - source
    return eq_ad, eq, flux_tpfa, dof_manager

eq_ad, eq_non_ad, flux_tpfa, dof_manager = setup_diff_problem(gb, "tp")
if use_ad:
    eq = eq_ad
else:
    eq = eq_non_ad
flux_tpfa.discretize(gb)
errors = []
for i in range(nsteps):
    for g, d in gb:
        p_loc = dof_manager.assemble_variable([g], variable, from_iterate=True)[
            dof_manager.grid_and_variable_to_dofs(g, variable)]
        # p_loc = dof_manager.assemble_variable([g], variable)[dof_manager.grid_and_variable_to_dofs(g, variable)]
        new_perm_value = permeability_from_pressure(p_loc)
        d[pp.PARAMETERS][keyword]["second_order_tensor"] = pp.SecondOrderTensor(
            new_perm_value
        )

    # Enforce rediscretization
    flux_tpfa.discretize(gb)
    arr = eq.evaluate(dof_manager)
    A, b = arr.jac, arr.val
    dx = pardisosolve(A, -b)
    dof_manager.distribute_variable(dx, grid_list, to_iterate=True, additive=True)
    print(f"b {np.linalg.norm(b)} and dx {np.linalg.norm(dx)}")
    p_num = dof_manager.assemble_variable(grid_list, from_iterate=True)
    errors.append(np.linalg.norm(p_num - p_analytical(x)))

p_num = dof_manager.assemble_variable(grid_list, from_iterate=True)
print("Use ad:", use_ad, "\nNorm of p_numerical - p_analytical:", np.linalg.norm(p_num-p_analytical(x)))
if use_ad:
    errors_ad = errors.copy()
else:
    errors_non_ad = errors.copy()
def plot(e, e_ad, nsteps):
    es = np.vstack((e, e_ad))
    iters = np.arange(nsteps)
    plt.semilogy(iters, e, label="non ad")
    plt.semilogy(iters, e_ad, label="ad")
    ax = plt.gca()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error (not normalized)")
    plt.legend()
if "errors_ad" in locals() and "errors_non_ad" in locals():
    plt.close("all")
    plot(errors_non_ad, errors_ad, nsteps)