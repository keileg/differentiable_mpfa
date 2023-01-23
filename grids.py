"""
Grid generation functions.

Each function constructs and returns a mixed-dimensional grid and a (unitary) domain dictionary.
"""


from typing import Any, Dict, List, Tuple

import numpy as np
import porepy as pp


def one_dimensional_grid_bucket(params: Dict) -> Tuple[pp.MixedDimensionalGrid, Dict]:
    phys_dims = np.asarray(params.get("phys_dims", [1]))
    n_cells = np.asarray(params.get("n_cells", [2]))
    box: Dict = pp.geometry.bounding_box.from_points(np.array([[0], phys_dims]).T)
    g: pp.Grid = pp.CartGrid(n_cells, physdims=phys_dims)
    mdg: pp.MixedDimensionalGrid = pp.meshing._assemble_mdg([[g]])
    return mdg, box


def two_dimensional_cartesian(params: Dict) -> Tuple[pp.MixedDimensionalGrid, Dict]:
    if params.get("simplex", False):
        box = pp.grids.standard_grids.utils.unit_domain(2)
        mdg = pp.grids.standard_grids.utils.make_mdg_2d_simplex(
            mesh_args=params["mesh_args"], points=None, fractures=None, domain=box
        )
        return mdg, box

    phys_dims = np.asarray(params.get("phys_dims", [1, 1]))
    n_cells = np.asarray(params.get("n_cells", [2, 2]))
    box: Dict = pp.geometry.bounding_box.from_points(np.array([[0, 0], phys_dims]).T)
    g: pp.Grid = pp.CartGrid(n_cells, physdims=phys_dims)
    mdg, _ = pp.meshing._assemble_mdg([[g]])
    return mdg, box


def three_dimensional_cartesian(params: Dict) -> Tuple[pp.MixedDimensionalGrid, Dict]:
    if params.get("simplex", False):
        raise NotImplementedError("Simplex grids not implemented for 3d.")

    phys_dims: List = params.get("phys_dims", [1, 1, 1])
    n_cells: List = params.get("n_cells", [2, 2, 2])
    box: Dict = pp.geometry.bounding_box.from_points(np.array([[0, 0, 0], phys_dims]).T)
    g: pp.Grid = pp.CartGrid(n_cells, physdims=phys_dims)
    mdg: pp.MixedDimensionalGrid = pp.meshing._assemble_mdg([[g]])[0]
    return mdg, box


def two_dimensional_cartesian_perturbed(
    params: Dict,
) -> Tuple[pp.MixedDimensionalGrid, Dict]:
    phys_dims = np.asarray(params.get("phys_dims", [1, 1]))
    n_cells = np.asarray(params.get("n_cells", [2, 2]))
    box: Dict = pp.geometry.bounding_box.from_points(np.array([[0, 0], phys_dims]).T)
    g: pp.Grid = pp.CartGrid(n_cells, physdims=phys_dims)
    internal_nodes = np.logical_not(g.tags["domain_boundary_nodes"])
    np.random.seed(0)
    g.nodes[0, internal_nodes] += (
        np.random.rand(internal_nodes.sum()) * phys_dims[0] / n_cells[0] / 3
    )
    g.nodes[0, internal_nodes] += phys_dims[0] / n_cells[0] / 6
    g.nodes[1, internal_nodes] += (
        np.random.rand(internal_nodes.sum()) * phys_dims[1] / n_cells[1] / 3
    )
    g.nodes[1, internal_nodes] += phys_dims[1] / n_cells[1] / 6

    mdg: pp.MixedDimensionalGrid = pp.meshing._assemble_mdg([[g]])[0]
    return mdg, box


def horizontal_fracture_2d(params: Dict) -> Tuple[pp.MixedDimensionalGrid, Dict]:
    endpoints = params.get("fracture_endpoints", [0, 1])
    mesh_args = params.get("mesh_args", [2, 2])
    simplex = params.get("simplex", False)
    return pp.md_grids_2d.single_horizontal(
        x_endpoints=endpoints, mesh_args=mesh_args, simplex=simplex
    )


def horizontal_fracture_3d(params):
    simplex = params.get("simplex", False)
    mesh_args = params.get("mesh_args", [2, 2, 2])
    endpoints = params.get("fracture_endpoints", [0.0, 1])
    return pp.md_grids_3d.single_horizontal(
        mesh_args=mesh_args,
        simplex=simplex,
        x_coords=endpoints,
        y_coords=endpoints,
    )


class Geometry(pp.models.geometry.ModelGeometry):
    """Mixin class for geometry."""

    params: dict[str, Any]
    """Parameters for the model."""

    def set_md_grid(self) -> None:
        """Create the mixed-dimensional grid.

        A unit square grid with no fractures is assigned by default.

        The method assigns the following attributes to self:
            mdg (pp.MixedDimensionalGrid): The produced mixed-dimensional grid.
            box (dict): The bounding box of the domain, defined through minimum and
                maximum values in each dimension.
        """
        mdg, domain = self.params["grid_method"](self.params)
        self.domain_bounds: dict = domain
        self.mdg: pp.MixedDimensionalGrid = mdg
        self.mdg.compute_geometry()
        self.nd = self.mdg.dim_max()
