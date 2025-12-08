import numpy as np
import scipy.sparse.linalg
from mpi4py import MPI
from dolfinx import mesh, fem
import dolfinx.fem.petsc
import ufl
from petsc4py import PETSc
import pgfenicsx as pg

# for the problem description, see https://fenicsproject.discourse.group/t/petrov-gelerkin-formulations-and-fenicsx/18369

# define mesh
msh = mesh.create_unit_interval(MPI.COMM_WORLD, 5)
msh.topology.create_connectivity(msh.topology.dim-1, msh.topology.dim)
tdim = msh.topology.dim-1

# define problem
x = ufl.SpatialCoordinate(msh)
n  = ufl.FacetNormal(msh)

u_exact = lambda x: np.sin(np.pi * x[0])
b = u_exact([0])
c = np.pi * np.cos(np.pi*0)
f  = ufl.pi**2 * ufl.sin(ufl.pi * x[0])

# get the two boundary parts {0} and {1}
gamma0_facets = mesh.locate_entities_boundary(msh, dim=tdim, marker=lambda tx: np.isclose(tx[0], 0.0))  # Gamma_0 = {0}
gamma1_facets = mesh.locate_entities_boundary(msh, dim=tdim, marker=lambda tx: np.isclose(tx[0], 1.0))  # Gamma_1 = {1}

# construct the boundary measure over gamma_0
gamma0 = mesh.meshtags(msh, tdim, gamma0_facets, np.full_like(gamma0_facets, 0))
ds = ufl.Measure("ds", domain=msh, subdomain_data=gamma0)
ds_gamma0 = ds(0)

# define the two (!) function spaces -> Petrov-Galerkin
U = fem.functionspace(msh, ("Lagrange", 1))
V = fem.functionspace(msh, ("Lagrange", 1))

# setup the different (!) Dirichlet BCs for each space
gamma0_dofs_U = fem.locate_dofs_topological(U, tdim, gamma0_facets)
gamma1_dofs_V = fem.locate_dofs_topological(V, tdim, gamma1_facets)
bcs = [fem.dirichletbc(b,   gamma0_dofs_U, U),
       fem.dirichletbc(0.0, gamma1_dofs_V, V)]

# define variational problem
u = ufl.TrialFunction(U)
v = ufl.TestFunction(V)

A = ufl.dot(ufl.grad(u),ufl.grad(v)) * ufl.dx
l = f*v*ufl.dx - c*v*ds_gamma0

A = fem.form(A)
l = fem.form(l)

# solve using pgfenicsx assembly routines (SciPy)
A_scipy = fem.assemble_matrix(A).to_scipy()
l_scipy = fem.assemble_vector(l).array

A_scipy_bcs = pg.apply_dirichletbc_matrix(A_scipy,U,V,bcs)
l_scipy_bcs = pg.apply_dirichletbc_vector(l_scipy,U,V,bcs)

u_scipy = fem.Function(U)
u_scipy.x.array[:] = scipy.sparse.linalg.spsolve(A_scipy_bcs,l_scipy_bcs)

# solve using pgfenicsx assembly routines (PETSc)
A_petsc = fem.petsc.assemble_matrix(A)
l_petsc = fem.petsc.assemble_vector(l)

A_petsc_bcs = pg.apply_dirichletbc_matrix(A_petsc,U,V,bcs)
l_petsc_bcs = pg.apply_dirichletbc_vector(l_petsc,U,V,bcs)

solver = PETSc.KSP().create(MPI.COMM_WORLD)
solver.setOperators(A_petsc_bcs)
solver.setType("preonly")
solver.getPC().setType("qr")
solver.setFromOptions()

u_petsc = fem.Function(U)
u_petsc.x.petsc_vec.ghostUpdate(
    addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
)
solver.solve(l_petsc_bcs, u_petsc.x.petsc_vec)

# compute error
u_exact_ = fem.Function(U)
u_exact_.interpolate(u_exact)
error_scipy = np.linalg.norm(u_scipy.x.array - u_exact_.x.array, ord=np.inf)
error_petsc = np.linalg.norm(u_petsc.x.array - u_exact_.x.array, ord=np.inf)
print(f"Error (SciPy): {error_scipy}")
print(f"Error (PETSc): {error_petsc}")



