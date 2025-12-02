import numpy as np
import pytest

from numbers import Number
from mpi4py import MPI
from dolfinx.mesh import create_unit_square, locate_entities_boundary, exterior_facet_indices
import dolfinx.fem as fem

from pgfenicsx import merge_dirichletbcs

dtype = np.float64

mesh = create_unit_square(MPI.COMM_WORLD, 4, 4, dtype=dtype)
tdim = mesh.topology.dim-1
mesh.topology.create_connectivity(tdim, mesh.topology.dim)

U = fem.functionspace(mesh, ("Lagrange", 1))
V = fem.functionspace(mesh, ("Lagrange", 2))
# using DG: da stimmt was mit den dofs nicht, checkn wie das dort läuft



def check_bcs(bc1, bc2):
    if isinstance(bc1, list):
        assert isinstance(bc2, list)
        assert len(bc1) == len(bc2)
        for b1,b2 in zip(bc1,bc2):
            check_bcs(b1,b2) 
        return
    idx1 = np.argsort(bc1.dof_indices()[0])
    idx2 = np.argsort(bc2.dof_indices()[0])
    assert np.array_equal(bc1.dof_indices()[0][idx1], bc2.dof_indices()[0][idx2])
    assert np.allclose(bc1.value.x.array[bc1.dof_indices()[0][idx1]], bc2.value.x.array[bc2.dof_indices()[0][idx2]])
    
def test_merge_Function_Function():
    u1_marker = lambda x: np.isclose(x[0], 0.0)
    u2_marker = lambda x: np.isclose(x[1], 0.0)
    u12_marker = lambda x: np.logical_or(u1_marker(x), u2_marker(x))

    u1_fun = lambda x: x[1]
    u2_fun = lambda x: x[0]**2
    u12_fun = lambda x: u1_marker(x) * u1_fun(x) + u2_marker(x) * u2_fun(x)

    u1_facets = locate_entities_boundary(mesh, dim=tdim, marker=u1_marker)
    u2_facets = locate_entities_boundary(mesh, dim=tdim, marker=u2_marker)
    u12_facets = locate_entities_boundary(mesh, dim=tdim, marker=u12_marker)

    u1 = fem.Function(U, dtype=dtype)
    u1.interpolate(u1_fun)
    u2 = fem.Function(U, dtype=dtype)
    u2.interpolate(u2_fun)
    u12 = fem.Function(U, dtype=dtype)
    u12.interpolate(u12_fun)
    
    bcs_U = [ fem.dirichletbc(u1, fem.locate_dofs_topological(U,tdim,u1_facets)),
              fem.dirichletbc(u2, fem.locate_dofs_topological(U,tdim,u2_facets))]
    bc_U_exact = fem.dirichletbc(u12, fem.locate_dofs_topological(U,tdim,u12_facets))._cpp_object

    check_bcs(merge_dirichletbcs(bcs_U, U), bc_U_exact)
    check_bcs(merge_dirichletbcs(bcs_U)[0], bc_U_exact)
    

def test_merge_Function_Constant():
    u1_marker = lambda x: np.isclose(x[0], 0.0)
    u2_marker = lambda x: np.isclose(x[1], 0.0)
    u12_marker = lambda x: np.logical_or(u1_marker(x), u2_marker(x))

    u1_fun = lambda x: x[1]
    u2_fun = lambda x: 0*x[0] # = ZERO
    u12_fun = lambda x: u1_marker(x) * u1_fun(x) + u2_marker(x) * u2_fun(x)

    u1_facets = locate_entities_boundary(mesh, dim=tdim, marker=u1_marker)
    u2_facets = locate_entities_boundary(mesh, dim=tdim, marker=u2_marker)
    u12_facets = locate_entities_boundary(mesh, dim=tdim, marker=u12_marker)

    u1 = fem.Function(U, dtype=dtype)
    u1.interpolate(u1_fun)
    u2 = fem.Function(U, dtype=dtype)
    u2.interpolate(u2_fun)
    u12 = fem.Function(U, dtype=dtype)
    u12.interpolate(u12_fun)
    
    ZERO = fem.Constant(mesh, dtype(0))
    bcs_U = [ fem.dirichletbc(u1,   fem.locate_dofs_topological(U,tdim,u1_facets)),
              fem.dirichletbc(ZERO, fem.locate_dofs_topological(U,tdim,u2_facets), U)]
    bc_U_exact = fem.dirichletbc(u12, fem.locate_dofs_topological(U,tdim,u12_facets))._cpp_object
    
    check_bcs(merge_dirichletbcs(bcs_U, U), bc_U_exact)
    check_bcs(merge_dirichletbcs(bcs_U)[0], bc_U_exact)
    
    
def test_empty_bcs():
    u_marker = lambda x: np.isclose(x[0], np.inf)
    u_fun = lambda x: x[1]
    u_facets = locate_entities_boundary(mesh, dim=tdim, marker=u_marker)
    
    u = fem.Function(U, dtype=dtype)
    u.interpolate(u_fun)
    
    bc_U_exact = fem.dirichletbc(u, fem.locate_dofs_topological(U,tdim,u_facets))._cpp_object
    check_bcs(merge_dirichletbcs(None,U), bc_U_exact)
    

def test_two_spaces_only_one_wanted():
    u1_marker = lambda x: np.isclose(x[0], 0.0)
    u2_marker = lambda x: np.isclose(x[1], 0.0)
    u12_marker = lambda x: np.logical_or(u1_marker(x), u2_marker(x))
    
    v1_marker = lambda x: np.isclose(x[0], 1.0)
    v2_marker = lambda x: np.isclose(x[1], 1.0)
    v12_marker = lambda x: np.logical_or(v1_marker(x), v2_marker(x))

    u1_fun = lambda x: x[1]
    u2_fun = lambda x: x[0]**2
    u12_fun = lambda x: u1_marker(x) * u1_fun(x) + u2_marker(x) * u2_fun(x)
    v1_fun = lambda x: (x[1]-1)
    v2_fun = lambda x: -(x[0]-1)**2
    v12_fun = lambda x: v1_marker(x) * v1_fun(x) + v2_marker(x) * v2_fun(x)
    
    u1_facets = locate_entities_boundary(mesh, dim=tdim, marker=u1_marker)
    u2_facets = locate_entities_boundary(mesh, dim=tdim, marker=u2_marker)
    u12_facets = locate_entities_boundary(mesh, dim=tdim, marker=u12_marker)
    
    v1_facets = locate_entities_boundary(mesh, dim=tdim, marker=v1_marker)
    v2_facets = locate_entities_boundary(mesh, dim=tdim, marker=v2_marker)
    v12_facets = locate_entities_boundary(mesh, dim=tdim, marker=v12_marker)

    u1 = fem.Function(U, dtype=dtype)
    u1.interpolate(u1_fun)
    u2 = fem.Function(U, dtype=dtype)
    u2.interpolate(u2_fun)
    u12 = fem.Function(U, dtype=dtype)
    u12.interpolate(u12_fun)
    
    v1 = fem.Function(V, dtype=dtype)
    v1.interpolate(v1_fun)
    v2 = fem.Function(V, dtype=dtype)
    v2.interpolate(v2_fun)
    v12 = fem.Function(V, dtype=dtype)
    v12.interpolate(v12_fun)
    
    bcs = [ fem.dirichletbc(u1, fem.locate_dofs_topological(U,tdim,u1_facets)),
            fem.dirichletbc(u2, fem.locate_dofs_topological(U,tdim,u2_facets)),
            fem.dirichletbc(v1, fem.locate_dofs_topological(V,tdim,v1_facets)),
            fem.dirichletbc(v2, fem.locate_dofs_topological(V,tdim,v2_facets))]
    
    
    bc_U_exact = fem.dirichletbc(u12, fem.locate_dofs_topological(U,tdim,u12_facets))._cpp_object
    bc_V_exact = fem.dirichletbc(v12, fem.locate_dofs_topological(V,tdim,v12_facets))._cpp_object
    
    check_bcs(merge_dirichletbcs(bcs, U), bc_U_exact)
    check_bcs(merge_dirichletbcs(bcs, V), bc_V_exact)
    check_bcs(merge_dirichletbcs(bcs,[U,V]), [bc_U_exact, bc_V_exact])
    check_bcs(merge_dirichletbcs(bcs,[V,U]), [bc_V_exact, bc_U_exact])
    check_bcs(merge_dirichletbcs(bcs), [bc_U_exact, bc_V_exact])
    
    
if __name__ == "__main__":
    test_merge_Function_Function()
    test_merge_Function_Constant()
    test_empty_bcs()
    test_two_spaces_only_one_wanted()
    