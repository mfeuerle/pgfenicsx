# Moritz Feuerle, 2025
r"""
This is an add-on for FEniCSx (developed using v0.10) to add support for Petrov-Galerkin variational formulations, i.e. variational formulations where the trial and test function spaces differ.

This module provides the two functions :func:`apply_dirichletbc_matrix` and :func:`apply_dirichletbc_vector` (and the combination :func:`apply_dirichletbc_system`) as a addon to handle Dirichlet boundary conditions in context of Petrov-Galerkin variational formulations. 

A example workflow for solving Petrov-Galerkin variational formulations could read as follows:

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
>>> A_ufl = ... # bilinear form using u and v
>>> l_ufl = ... # linear form using v
>>> A_form = fem.form(A_ufl)
>>> l_form = fem.form(l_ufl)

Now, the system can be assembled into SciPy or PETSc, bevor using this module to apply the boundary conditions:

>>> # Using scipy:
>>> A = fem.assemble_matrix(A_form).to_scipy() # dont use bcs here!
>>> l = fem.assemble_vector(l_form).array      # dont call fem.apply_lifting!

or alternatively:

>>> # Using PETSc:
>>> import dolfinx.fem.petsc
>>> A = dolfinx.fem.petsc.assemble_matrix(A_form) # dont use bcs here!
>>> l = dolfinx.fem.petsc.assemble_vector(l_form) # dont call fem.petsc.apply_lifting!

Either way, the Dirichlet boundary conditions can now be applied using this module:

>>> import pgfenicsx as pg
>>> A_bcs = pg.apply_dirichletbc_matrix(A,U,V,bcs)
>>> l_bcs = pg.apply_dirichletbc_vector(l,U,V,bcs)

Finally, solving the linear systems can be done as usual, altough - for Petrov-Galerkin formulations - the system matrix might not be square. In this case, a least squares solver such as the QR decomposition needs to be used:

>>> # Using scipy:
>>> from scipy.sparse.linalg import lsqr
>>> u = fem.Function(U)
>>> u.x.array[:] = lsqr(A_bcs,l_bcs)[0] 

>>> # Using PETSc:
>>> from petsc4py import PETSc
>>> solver = PETSc.KSP().create(MPI.COMM_WORLD)
>>> solver.setOperators(A_bcs)
>>> solver.setType("preonly")
>>> solver.getPC().setType("qr")
>>> solver.setFromOptions()
>>> u = fem.Function(U)
>>> u.x.petsc_vec.ghostUpdate(
         addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
>>> )
>>> solver.solve(l_bcs, u.x.petsc_vec)
   
Functions
-----------------
.. autosummary::
   :toctree: _generated/ 
   
   apply_dirichletbc_system
   apply_dirichletbc_vector
   apply_dirichletbc_matrix
   merge_dirichletbcs


.. sectionauthor:: Moritz Feuerle, 2022
"""
# generating the list of functions / classes / etc. above automatically might be bossible with https://stackoverflow.com/a/18143318


from ._pgfenicsx import *

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
