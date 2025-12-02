# Moritz Feuerle, 2025
r"""
This is an add-on for FEniCSx (developed using v0.10) to add support for Petrov-Galerkin variational formulations, i.e. variational formulations where the trial and test function spaces differ.

This module provides the two functions :func:`assemble_matrix` and :func:`assemble_vector` as a drop-in replacement for :func:`dolfinx.fem.assemble_matrix` / :func:`dolfinx.fem.petsc.assemble_matrix` and :func:`dolfinx.fem.assemble_vector` / :func:`dolfinx.fem.petsc.assemble_vector` to assemble Petrov-Galerkin variational formulations. Thus, 

>>> A_ = pgfenicsx.assemble_matrix(A, bcs, petsc=False,...)
>>> A_ = pgfenicsx.assemble_matrix(A, bcs, petsc=True,...)

replace

>>> A_ = dolfinx.fem.assemble_matrix(A, bcs,...).toscipy()
>>> A_ = dolfinx.fem.petsc.assemble_matrix(A, bcs,...)
   
respectively, while

>>> l_ = pgfenicsx.assemble_vector(l, trial_space, bcs, petsc=False,...)
>>> l_ = pgfenicsx.assemble_vector(l, trial_space, bcs, petsc=True,...)

replace

>>> l_ = dolfinx.fem.assemble_vector(l, bcs,...).toscipy()
>>> l_ = dolfinx.fem.petsc.assemble_vector(l, bcs,...)
   
respectively.

A complete workflow for solving Petrov-Galerkin variational formulations could read as follows:

Create mesh, function spaces, trial and test functions, variational forms, etc. as usual using dolfinx, altough using different function spaces for trial and test functions as well as defining different Dirichlet boundary conditions on these function spaces, e.g.,:

>>> from dolfinx import fem, mesh
>>> # define different trial and test spaces
>>> U = fem.functionspace(...)
>>> V = fem.functionspace(...)
>>> # identify boundary dofs
>>> dofs_U = fem.locate_dofs_topological(U,...) # or fem.locate_dofs_geometrical(U,...)
>>> dofs_V = fem.locate_dofs_topological(V,...) # or fem.locate_dofs_geometrical(V,...)
>>> # define trial boundary function
>>> uD = fem.Function(U)
>>> uD.interpolate(...)
>>> # define Dirichlet BCs
>>> bcs = [fem.dirichletbc(uD, dofs_U),     # trial bc
           fem.dirichletbc(0.0, dofs_V, V)] # test bc
>>> # define the variational system
>>> u = ufl.TrialFunction(U)
>>> v = ufl.TestFunction(V)
>>> A = ... # bilinear form using u and v
>>> l = ... # linear form using v

Now, this module comes into play for assembling the system matrix and vector as follows:

>>> # Using scipy:
>>> import pgfenicsx
>>> A_scipy = pgfenicsx.assemble_matrix(fem.form(A), bcs, petsc=False)
>>> l_scipy = pgfenicsx.assemble_vector(fem.form(l), U, bcs, petsc=False)

>>> # Using PETSc:
>>> import pgfenicsx
>>> A_petsc = pgfenicsx.assemble_matrix(fem.form(A), bcs, petsc=True)
>>> l_petsc = pgfenicsx.assemble_vector(fem.form(l), U, bcs, petsc=True)

Finally, solving the linear systems can be done as usual, altough - for Petrov-Galerkin formulations - the system matrix might not be square. In this case, a least squares solver such as the QR decomposition needs to be used:

>>> # Using scipy:
>>> from scipy.sparse.linalg import lsqr
>>> u_scipy = fem.Function(U)
>>> u_scipy.x.array[:] = lsqr(A_scipy,l_scipy)[0] 

>>> # Using PETSc:
>>> from petsc4py import PETSc
>>> solver = PETSc.KSP().create(MPI.COMM_WORLD)
>>> solver.setOperators(A_petsc)
>>> solver.setType("preonly")
>>> solver.getPC().setType("qr")
>>> solver.setFromOptions()
>>> u_petsc = fem.Function(U)
>>> u_petsc.x.petsc_vec.ghostUpdate(
         addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
>>> )
>>> solver.solve(l_petsc, u_petsc.x.petsc_vec)
   
Functions
-----------------
.. autosummary::
   :toctree: _generated/ 
   
   assemble_matrix
   assemble_vector
   merge_dirichletbcs


.. sectionauthor:: Moritz Feuerle, 2022
"""
# generating the list of functions / classes / etc. above automatically might be bossible with https://stackoverflow.com/a/18143318


from ._pgfenicsx import *
from . import utils

# prevent that all that clutter below end up in __all__
__all__ = [s for s in dir() if not s.startswith('_')]



# Hacky workaround for TypeAlias not being shown in the docs
# Might be fixed in Sphinx v9.0, see the upcoming option .. autotype:: and https://github.com/sphinx-doc/sphinx/pull/13808
# from typing import TypeAlias as _TypeAlias
# from typing import TypeAliasType as _TypeAliasType

# import dolfinx

# for x in dir():
#    if isinstance(eval(x),_TypeAliasType):
#       exec("%s: _TypeAlias = %s" % (x,str(eval(x).__value__)))
