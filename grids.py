import porepy as pp
import numpy as np
from typing import Dict, List


def one_dimensional_grid_bucket(params):
    phys_dims: List = params.get("phys_dims", [1])
    n_cells: List = params.get("n_cells", [2])
    box: Dict = pp.geometry.bounding_box.from_points(np.array([[0], phys_dims]).T)
    g: pp.Grid = pp.CartGrid(n_cells, physdims=phys_dims)
    gb: pp.GridBucket = pp.meshing._assemble_in_bucket([[g]])
    return gb, box


def two_dimensional_cartesian(params):
    if params.get("simplex", False):
        domain = pp.grids.standard_grids.utils.unit_domain(2)
        gb = pp.grids.standard_grids.utils.make_gb_2d_simplex(
            mesh_args=params["mesh_args"], points=None, edges=None, domain=domain
        )
        return gb, domain

    phys_dims: List = params.get("phys_dims", [1, 1])
    n_cells: List = params.get("n_cells", [2, 2])
    box: Dict = pp.geometry.bounding_box.from_points(np.array([[0, 0], phys_dims]).T)
    g: pp.Grid = pp.CartGrid(n_cells, physdims=phys_dims)
    gb: pp.GridBucket = pp.meshing._assemble_in_bucket([[g]])
    return gb, box


def two_dimensional_cartesian_perturbed(params):
    phys_dims: List = params.get("phys_dims", [1, 1])
    n_cells: List = params.get("n_cells", [2, 2])
    box: Dict = pp.geometry.bounding_box.from_points(np.array([[0, 0], phys_dims]).T)
    g: pp.Grid = pp.CartGrid(n_cells, physdims=phys_dims)
    internal_nodes = np.logical_not(g.tags["domain_boundary_nodes"])
    g.nodes[0, internal_nodes] += (
        np.random.rand(internal_nodes.sum()) * phys_dims[0] / n_cells[0] / 4
    )
    g.nodes[1, internal_nodes] += (
        np.random.rand(internal_nodes.sum()) * phys_dims[1] / n_cells[1] / 3
    )
    gb: pp.GridBucket = pp.meshing._assemble_in_bucket([[g]])
    return gb, box


def horizontal_fracture_2d(params):
    endpoints = params.get("fracture_endpoints", [0, 1])
    mesh_args = params.get("mesh_args", [2, 2])
    simplex = params.get("simplex", False)
    return pp.grid_buckets_2d.single_horizontal(
        x_endpoints=endpoints, mesh_args=mesh_args, simplex=simplex
    )


def horizontal_fracture_3d(params):
    simplex = params.get("simplex", False)
    mesh_args = params.get("mesh_args", [2, 2, 2])
    return pp.grid_buckets_3d.single_horizontal(mesh_args=mesh_args, simplex=simplex)
