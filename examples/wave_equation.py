import numpy as np
from scipy import sparse
from mpi4py import MPI
from dolfinx import mesh, fem
import dolfinx.fem.petsc
import ufl
from petsc4py import PETSc
from enum import IntEnum
import pgfenicsx as pg

############################################
# Problem: space-time wave equation
############################################
# find u such that
#   - u_tt + u_xx = f  in I x Omega
#       u   = u0 on {t=0} x Omega
#       u_t = u1 on {t=0} x Omega
#       u   = g0 on I x Gamma   (Gamma = boundary of Omega)

############################################
# Variational formulation:
############################################
# Using one integration by parts in time and space, we get:
# Find u in U such that
#   (u_t, v_t)_L2(IxOmega) + (u_x, v_x)_L2(IxOmega) = (f,v)_L2(IxOmega) + (u1,v(0))_L2(Omega)  for all v in V
# with 
#   U = {u in H1(IxOmega) : u(0) = u0 in Omega, u=g0 on IxGamma}
#   V = {v in H1(IxOmega) : v(T) = 0  in Omega, v=0  on IxGamma}


############################################
# Create mesh and define problem parameters
############################################
I     = [0.0, 3.0]
Omega = [0.0, 2.0]

nx = 20
nt = np.ceil((I[1]-I[0])/((Omega[1]-Omega[0])/(nx+1))).astype('int')    # ensure CFL condition

msh = mesh.create_rectangle(MPI.COMM_WORLD, [[I[0], Omega[0]], [I[1], Omega[1]]], [nt, nx])
tdim = msh.topology.dim
msh.topology.create_connectivity(tdim-1, tdim)

tx = ufl.SpatialCoordinate(msh)
n  = ufl.FacetNormal(msh)

u_exact = lambda tx: np.sin(np.pi*tx[0])*tx[1] + 1  # exact solution as reference
f  = - ufl.pi**2 * ufl.sin(ufl.pi*tx[0])*tx[1]      # right-hand side
u0 = u_exact                                        # dirichlet BC at initial time    
u1 = ufl.pi * ufl.cos(ufl.pi*tx[0])*tx[1]           # neuman BC at initial time
g0 = u_exact                                        # dirichlet BC at spatial boundary

############################################
# Define function spaces and boundary parts
############################################
U = fem.functionspace(msh, ("Lagrange", 1))
V = fem.functionspace(msh, ("Lagrange", 1))

class bdry_prt(IntEnum): t0,T,gamma = range(3)
bdry  = [(bdry_prt.t0,    lambda tx: np.isclose(tx[0], I[0])),  # initial time
         (bdry_prt.T,     lambda tx: np.isclose(tx[0], I[1])),  # terminal time
         (bdry_prt.gamma, lambda tx: np.logical_or(np.isclose(tx[1], Omega[0]),np.isclose(tx[1], Omega[1])))]  # I x Gamma

############################################
# Setup boundary parts
############################################
facets = np.empty((0,), dtype=np.int32)
tags   = np.empty((0,), dtype=np.int32)
for b in bdry:
    facets_b = mesh.locate_entities_boundary(msh, dim=tdim-1, marker=b[1])
    facets = np.append(facets, facets_b)
    tags   = np.append(tags,   np.full_like(facets_b, b[0], dtype=np.int32))

bdry_tagged = mesh.meshtags(msh, tdim-1, facets, tags)
ds = ufl.Measure("ds", domain=msh, subdomain_data=bdry_tagged)

############################################
# Setup Dirichlet BCs
############################################
def get_dofs(space, bdry_part):
    return fem.locate_dofs_topological(space, tdim-1, bdry_tagged.find(bdry_part))

u0_U = fem.Function(U)
u0_U.interpolate(u0)
g0_U = fem.Function(U)
g0_U.interpolate(g0)

bcs = [ fem.dirichletbc(u0_U, get_dofs(U, bdry_prt.t0)      ),
        fem.dirichletbc(g0_U, get_dofs(U, bdry_prt.gamma)   ),
        fem.dirichletbc(0.0,  get_dofs(V, bdry_prt.T),     V),
        fem.dirichletbc(0.0,  get_dofs(V, bdry_prt.gamma), V)]

bcs = pg.merge_dirichletbcs(bcs)

############################################
# Setup variational problem
############################################
u = ufl.TrialFunction(U)
v = ufl.TestFunction(V)

A =  - ufl.inner(ufl.grad(u)[0],ufl.grad(v)[0]) * ufl.dx
for i in range(1,len(tx)):
    A += ufl.inner(ufl.grad(u)[i],ufl.grad(v)[i]) * ufl.dx

l = f*v*ufl.dx + u1*v*ds(bdry_prt.t0)

A = fem.form(A)
l = fem.form(l)

############################################
# Interpolate exact solution
############################################
u_exact_ = fem.Function(U)
u_exact_.interpolate(u_exact)

############################################
# Solve the variational problem using SciPy
#############################################
A_scipy = fem.assemble_matrix(A).to_scipy()
l_scipy = fem.assemble_vector(l).array

A_scipy_bcs = pg.apply_dirichletbc_matrix(A_scipy,U,V,bcs)
l_scipy_bcs = pg.apply_dirichletbc_vector(l_scipy,U,V,bcs)

u_scipy = fem.Function(U)
u_scipy.x.array[:] = sparse.linalg.spsolve(A_scipy_bcs,l_scipy_bcs)

print(f"Error (SciPy): {np.linalg.norm(u_scipy.x.array - u_exact_.x.array, ord=np.inf)}")

############################################
# Solve the variational problem using PETSc
#############################################
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

print(f"Error (PETSc): {np.linalg.norm(u_petsc.x.array - u_exact_.x.array, ord=np.inf)}")



############################################
# Visualise the solution using pyvista
#############################################   
try:
    import pyvista
    from dolfinx import plot
    from pathlib import Path
    
    results_folder = Path(__file__).parent / f'plots_{Path(__file__).stem}'
    
    def plot_pyvista(u,name, plotter):
        cells, types, x = plot.vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(cells, types, x)
        grid.point_data["u"] = u
        grid.set_active_scalars("u")
        grid.rotate_z(180, inplace=True)
        plotter.add_mesh(grid.warp_by_scalar(), show_edges=True)
        plotter.show_grid(xtitle='t', ytitle='x', ztitle='u(t,x)')
        plotter.add_text(name)
        
    
    plotter = pyvista.Plotter(shape=(1,3))
    plotter.subplot(0, 0)
    plot_pyvista(u_exact_.x.array, "u_exact", plotter)
    plotter.subplot(0, 1)
    plot_pyvista(u_scipy.x.array, "u_scipy", plotter)
    plotter.subplot(0, 2)
    plot_pyvista(u_petsc.x.array, "u_petsc", plotter)
    
    if pyvista.OFF_SCREEN:
        plotter.screenshot(results_folder / f".png")
    else:
        plotter.show()

except ModuleNotFoundError:
    print("'pyvista' is required to visualise the solution.")
    print("To install pyvista with pip: 'python3 -m pip install pyvista' or conda: 'conda install -c conda-forge pyvista'.")