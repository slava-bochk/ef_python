from typing import List

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from ef.inner_region import InnerRegion
from ef.meshgrid import MeshGrid


class FieldSolver:
    def __init__(self, mesh: MeshGrid, inner_regions: List[InnerRegion],
                 tolerance: float = 1e-10, max_iter: int = 1000):
        if inner_regions:
            print("WARNING: field-solver: inner region support is untested")
            print("WARNING: proceed with caution")
        self._double_index = self.double_index(mesh.n_nodes)
        self.mesh = mesh
        self.nodes_in_regions, self.potential_in_regions = self.generate_nodes_in_regions(inner_regions)
        nrows = (mesh.n_nodes - 2).prod()
        self.A = self.construct_equation_matrix()
        self.phi_vec = np.zeros(nrows)
        self.rhs = np.empty_like(self.phi_vec)
        self.tolerance = tolerance
        self.max_iter = max_iter

    def construct_equation_matrix(self):
        nx, ny, nz = self.mesh.n_nodes - 2
        size = nx * ny * nz
        cx, cy, cz = self.mesh.cell ** 2
        dx, dy, dz = cy * cz, cx * cz, cx * cy
        diag_dx = self.get_diag_d2dx2_in_3d(nx, ny, nz, dx)
        diag_dy = self.get_diag_d2dy2_in_3d(nx, ny, nz, dy)
        i = np.concatenate((np.arange(size),
                            np.arange(size - 1), np.arange(1, size),
                            np.arange(size - nx), np.arange(nx, size),
                            np.arange(size - nx * ny), np.arange(nx * ny, size)))
        j = np.concatenate((np.arange(size),
                            np.arange(1, size), np.arange(size - 1),
                            np.arange(nx, size), np.arange(size - nx),
                            np.arange(nx * ny, size), np.arange(size - nx * ny)))
        values = np.concatenate((np.full(size, -2.0 * (dx + dy + dz)),
                                 diag_dx, diag_dx,
                                 diag_dy, diag_dy,
                                 np.full(size - nx * ny, dz), np.full(size - nx * ny, dz)))
        matrix = scipy.sparse.coo_matrix((values, (i, j)), shape=(size, size))
        return self.zero_nondiag_for_nodes_inside_objects(matrix).tocsr()

    @staticmethod
    def get_diag_d2dx2_in_3d(nx, ny, nz, dx):
        diag_offset = 1
        block_size = nx
        ones = np.full(block_size - diag_offset, dx)
        zeros = np.zeros(diag_offset)
        return np.concatenate([ones, zeros] * (ny * nz))[:-diag_offset]

    @staticmethod
    def get_diag_d2dy2_in_3d(nx, ny, nz, dy):
        diag_offset = nx
        block_size = nx * ny
        ones = np.full(block_size - diag_offset, dy)
        zeros = np.zeros(diag_offset)
        return np.concatenate([ones, zeros] * nz)[:-diag_offset]

    def generate_nodes_in_regions(self, inner_regions):
        ijk = self._double_index[:, 1:]
        n = self._double_index[:, 0]
        xyz = self.mesh.cell * ijk
        inside = np.zeros_like(n, np.bool)
        potential = np.empty_like(n, np.float)
        for ir in inner_regions:
            mask = ir.check_if_points_inside(xyz)
            mask = mask.get() if hasattr(mask, 'get') else mask
            if np.logical_and.reduce([mask, inside, potential != ir.potential]).any():
                raise ValueError("Found intersecting inner regions with different potentials.")
            potential[mask] = ir.potential
            inside = np.logical_or(inside, mask)
        indices = n[inside]
        return indices, potential[indices]

    def zero_nondiag_for_nodes_inside_objects(self, matrix: scipy.sparse.coo_matrix):
        data = matrix.data
        row = matrix.row
        col = matrix.col
        mask = np.isin(row, self.nodes_in_regions)
        data[mask] = 0.
        data[np.logical_and(row == col, mask)] = 1.
        return scipy.sparse.coo_matrix((data, (row, col)), shape=matrix.shape)

    def eval_potential(self, charge_density, potential):
        raise NotImplementedError()

    def init_rhs_vector(self, charge_density, potential):
        self.init_rhs_vector_in_full_domain(charge_density, potential)
        self.set_rhs_for_nodes_inside_objects()

    def init_rhs_vector_in_full_domain(self, charge_density, potential):
        charge = charge_density.data
        pot = potential.data
        rhs = -4 * np.pi * self.mesh.cell.prod() ** 2 * charge[1:-1, 1:-1, 1:-1]
        dx, dy, dz = self.mesh.cell
        rhs[0] -= dy * dy * dz * dz * pot[0, 1:-1, 1:-1]
        rhs[-1] -= dy * dy * dz * dz * pot[-1, 1:-1, 1:-1]
        rhs[:, 0] -= dx * dx * dz * dz * pot[1:-1, 0, 1:-1]
        rhs[:, -1] -= dx * dx * dz * dz * pot[1:-1, -1, 1:-1]
        rhs[:, :, 0] -= dx * dx * dy * dy * pot[1:-1, 1:-1, 0]
        rhs[:, :, -1] -= dx * dx * dy * dy * pot[1:-1, 1:-1, -1]
        self.rhs = rhs.ravel('F')

    def set_rhs_for_nodes_inside_objects(self):
        self.rhs[self.nodes_in_regions] = self.potential_in_regions

    def transfer_solution_to_spat_mesh(self, potential):
        potential._data[1:-1, 1:-1, 1:-1] = potential.xp.asarray(self.phi_vec.reshape(self.mesh.n_nodes - 2, order='F'))

    @staticmethod
    def double_index(n_nodes):
        nx, ny, nz = n_nodes - 2
        i, j, k = np.mgrid[0:nx, 0:ny, 0:nz].reshape((3, -1), order='F')
        return np.column_stack((i + j * nx + k * nx * ny, i + 1, j + 1, k + 1))
