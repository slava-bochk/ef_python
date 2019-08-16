from typing import List

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from ef.inner_region import InnerRegion
from ef.meshgrid import MeshGrid


class FieldSolver:
    def __init__(self, mesh: MeshGrid, inner_regions: List[InnerRegion]):
        if inner_regions:
            print("WARNING: field-solver: inner region support is untested")
            print("WARNING: proceed with caution")
        self._double_index = self.double_index(mesh.n_nodes)
        self.mesh = mesh
        self.nodes_in_regions, self.potential_in_regions = self.generate_nodes_in_regions(inner_regions)
        nrows = (mesh.n_nodes - 2).prod()
        self.A = self.construct_equation_matrix()
        self.phi_vec = np.empty(nrows)
        self.rhs = np.empty_like(self.phi_vec)
        self.create_solver_and_preconditioner()

    def construct_equation_matrix(self):
        nx, ny, nz = self.mesh.n_nodes - 2
        cx, cy, cz = self.mesh.cell ** 2
        dx, dy, dz = cy * cz, cx * cz, cx * cy
        matrix = dx * self.construct_d2dx2_in_3d(nx, ny, nz) + \
                 dy * self.construct_d2dy2_in_3d(nx, ny, nz) + \
                 dz * self.construct_d2dz2_in_3d(nx, ny, nz)
        return self.zero_nondiag_for_nodes_inside_objects(matrix)

    @staticmethod
    def construct_d2dx2_in_3d(nx, ny, nz):
        diag_offset = 1
        block_size = nx
        block = scipy.sparse.diags([1.0, -2.0, 1.0], [-diag_offset, 0, diag_offset], shape=(block_size, block_size),
                                   format='coo')
        big_block = scipy.sparse.block_diag([block] * nz, format='coo')
        return scipy.sparse.block_diag([big_block] * ny, format='csr')

    @staticmethod
    def construct_d2dy2_in_3d(nx, ny, nz):
        diag_offset = nx
        block_size = nx * ny
        block = scipy.sparse.diags([1.0, -2.0, 1.0], [-diag_offset, 0, diag_offset], shape=(block_size, block_size),
                                   format='coo')
        return scipy.sparse.block_diag([block] * nz, format='csr')

    @staticmethod
    def construct_d2dz2_in_3d(nx, ny, nz):
        diag_offset = nx * ny
        block_size = nx * ny * nz
        return scipy.sparse.diags([1.0, -2.0, 1.0], [-diag_offset, 0, diag_offset], shape=(block_size, block_size),
                                  format='csr')

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

    def zero_nondiag_for_nodes_inside_objects(self, matrix):
        for i in self.nodes_in_regions:
            csr_row_start = matrix.indptr[i]
            csr_row_end = matrix.indptr[i + 1]
            for t in range(csr_row_start, csr_row_end):
                if matrix.indices[t] != i:
                    matrix.data[t] = 0
                else:
                    matrix.data[t] = 1
        return matrix

    def create_solver_and_preconditioner(self):
        raise NotImplementedError()

    def eval_potential(self, charge_density, potential):
        self.solve_poisson_eqn(charge_density, potential)

    def solve_poisson_eqn(self, charge_density, potential):
        raise NotImplementedError()

    def init_rhs_vector(self, charge_density, potential):
        self.init_rhs_vector_in_full_domain(charge_density, potential)
        self.set_rhs_for_nodes_inside_objects()

    def init_rhs_vector_in_full_domain(self, charge_density, potential):
        # TODO: use cupy with amgx?
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
        return np.array([(i + j * nx + k * nx * ny, i + 1, j + 1, k + 1)
                         for k in range(nz) for j in range(ny) for i in range(nx)])
