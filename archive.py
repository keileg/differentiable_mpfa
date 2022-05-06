import numpy as np
import porepy as pp

from models import BiotNonlinearTpfa
from run_chemistry import ChemistryParameters
from run_verification import SquareRootPermeability


class EquilibriumChemistryParameters(ChemistryParameters):
    def _set_chemistry_parameters(self) -> None:
        super()._set_chemistry_parameters()
        comp = "b"
        for g, d in self.gb:
            parameters = {"mass_weight": np.ones(g.num_cells),
                          "bc": getattr(self, "_bc_type_" + comp)(g),
                          "bc_values": getattr(self, "_bc_values_" + comp)(g),
                          "source": getattr(self, "_source_" + comp)(g),
                          "stoichiometric_coefficient": getattr(self, "_stoichiometric_coefficient_" + comp)(g),
                          }
            pp.initialize_data(g,
                               d,
                               getattr(self, comp + "_parameter_key"),
                               parameters,
                               )

    def _bc_type_b(self, g: pp.Grid) -> pp.BoundaryCondition:
        """Dirichlet conditions on all external boundaries."""
        # Define boundary regions
        all_bf, *_ = self._domain_boundary_sides(g)
        # Define boundary condition on faces
        return pp.BoundaryCondition(g, all_bf, "dir")

    def _bc_values_b(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous boundary values.

        Units:
            Dirichlet conditions: Pa = kg / m^1 / s^2
            Neumann conditions: m^3 / s
        """
        return np.zeros(g.num_faces)

    def _stoichiometric_coefficient_b(self, g: pp.Grid) -> int:
        """Stoichiometric coefficient for the chemical reaction

        Args:
            g: pp.Grid

        Returns:
            np.ndarray: (g.num_cells)

        """
        return self._beta * np.ones(g.num_cells)



    def _source_b(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous values except at the domain center."""

        return self._domain_center_source(g, 0.55) * g.cell_volumes

    def _initial_b(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous initial values.

        Args:
            g: pp.Grid of the subdomain.

        Returns:
            vals: np.ndarray of cell-wise concentration values
        """
        return 0.5 * np.ones(g.num_cells)


class ShearDilation(BiotNonlinearTpfa):
    def _bc_values_mechanics(self, g: pp.Grid) -> np.ndarray:
        vals = np.zeros((self._Nd, g.num_faces))
        faces = g.face_centers[self._Nd - 1] > 1 - 1e-7
        vals[self._Nd - 1, faces] = -1e-3
        vals[self._Nd - 2, faces] = 2e-3
        return vals.ravel(order="F")

    def _dilation_angle(self, g):
        return .2 * np.ones(g.num_cells)

    def _friction_coefficient(self, g):
        return 5e-1 * np.ones(g.num_cells)

    def _source_scalar(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous Dirichlet values except at left end of fracture."""
        vals = np.zeros(g.num_cells)
        if g.dim == self._Nd - 1:
            pt = np.zeros(3)
            ind = g.closest_cell(pt)
            val = 0.01
            if g.dim == 1:
                val = 8e-3
            vals[ind] = val
        return vals


class SquarePermeability2d(SquareRootPermeability):
    def _source(self, g: pp.Grid) -> np.ndarray:
        """Source term.

        Units: m^3 / s
        """
        x = g.cell_centers[0]
        y = g.cell_centers[1]
        val = (
                -2 * x * (1 - x) * (x ** 2 + 0.1)
                + 2 * x * (-x * y * (1 - y) + y * (1 - x) * (1 - y))
                - 2 * y * (1 - y) * (x ** 2 + 0.1)
        )
        return val * g.cell_volumes

    def p_analytical(self, g=None):
        if g is None:
            g = self.gb.grids_of_dimension(self.gb.dim_max())[0]
        x = g.cell_centers[0]
        y = g.cell_centers[1]
        return x * y * (1 - x) * (1 - y)

    def _permeability_function(self, pressure: pp.ad.Variable):
        val = pressure ** 2 + 0.1
        return val

    def _permeability_function_ad(self, pressure: pp.ad.Variable):
        val = pressure ** 2 + 0.1
        return val