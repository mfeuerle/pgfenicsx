# Moritz Feuerle, 2025

__all__ = ['DirichletBC', 'dirichletbc', 'collect_dirichletbcs', 'setup_boundary_meshtags', 'concatenate_meshtags', 'assemble_system', 'assemble_matrix', 'assemble_vector']


import numpy as np
import scipy.sparse as sparse
from numbers import Number
from dolfinx import default_scalar_type
from dolfinx.fem.function import FunctionSpace
from dolfinx import mesh, fem
import ufl
import dolfinx.fem.petsc
from petsc4py import PETSc

from typing import TypeAlias
from collections.abc import Iterable, Callable
BoundaryFunctionType: TypeAlias = fem.Function | fem.Constant | fem.Expression | Number | Callable


def as_numpy_vector(x, dtype=None):
    """Ensure that x is a numpy array of the given dtype and shape (n,)."""
    return np.asarray(x, dtype=dtype).reshape((-1,))

def setup_boundary_meshtags(msh: mesh.Mesh, boundary_parts: Iterable[tuple[int, Callable[[np.ndarray], np.ndarray[bool]] | np.ndarray[int] | None]]) -> mesh.MeshTags:
    
    tags = [part[0] for part in boundary_parts]
    tags_unique = np.unique(tags)
    tags_occurrences = [np.where(tags==tag)[0] for tag in tags_unique]
    
    tags_leftovers = []
    mesh_tags = np.empty((2, 0), np.int32)
    
    for i in range(len(tags_unique)):
        tag = tags_unique[i]
        facets = np.empty((1, 0), np.int32)
        for j in tags_occurrences[i]:
            boundary = boundary_parts[j]
            if len(boundary) == 1 or boundary[1] is None:
                tags_leftovers.append(tag)
                continue
            if isinstance(boundary[1], np.ndarray):
                facets = np.append(facets, boundary[1])
            else:
                facets = np.append(facets, mesh.locate_entities_boundary(msh, dim=msh.topology.dim-1, marker=boundary[1]))
        facets = np.unique(facets)
        mesh_tags = np.append(mesh_tags, np.vstack((facets, np.full_like(facets, tag))),axis=1)
        
    if len(tags_leftovers) > 0:
        tags_leftovers = np.unique(tags_leftovers)
        all_bdry_facets = mesh.exterior_facet_indices(msh.topology)
        facets = np.setdiff1d(all_bdry_facets,mesh_tags[0])
        for tag in tags_leftovers:
            mesh_tags = np.append(mesh_tags,np.vstack((facets,  np.full_like(facets, tag))),axis=1)
            
    mesh_tags = mesh_tags[:, np.argsort(mesh_tags[0])]
    return mesh.meshtags(msh, msh.topology.dim-1, mesh_tags[0], mesh_tags[1])


def concatenate_meshtags(mesh: mesh.Mesh, meshtags: Iterable[mesh.MeshTags]) -> mesh.MeshTags:
    values = np.empty((0,), dtype=np.int32)
    indices = np.empty((0,), dtype=np.int32)
    for mt in meshtags:
        values = np.append(values, mt.values)
        indices = np.append(indices, mt.indices)
    idx = np.argsort(indices)
    return mesh.meshtags(mesh, mesh.topology.dim-1, indices[idx],  values[idx])

class DirichletBC:
    r"""Class representing a Dirichlet boundary condition as a replacement of :class:`dolfinx.fem.DirichletBC`."""
    
    # @classmethod
    # def from_dolfinx(cls, bc: fem.DirichletBC):
    #     r"""Create a :class:`DirichletBC` from a :class:`dolfinx.fem.DirichletBC`.
        
    #     Args:
    #         bc:
    #             The :class:`dolfinx.fem.DirichletBC` to convert.
    #     """
    #     u = fem.Function(bc.function_space)
    #     u.x.array[:] = default_scalar_type(0)
    #     u.x.array[bc.dof_indices()] = bc.g.x.array[bc.dof_indices()]
    #     return cls(bc.function_space, u, bc.dof_indices())
    
    def __init__(self, function_space: FunctionSpace, u: BoundaryFunctionType | np.ndarray[float] | fem.DirichletBC, dofs: np.ndarray[int] = None):
        r"""Create a Dirichlet boundary condition restricting the ``dofs`` of ``function_space`` with the the values given by ``u``.
        
        Args:
            function_space:
                The function space the boundary condition is defined on.
            u:
                Object defining the dirichlet values at the given dofs. Can be given either the same way as in :func:`dirichletbc`, or as a discrete array directly specifying the values at each dof, i.e. an array of a) same length as ``dofs``, i.e. ``u[i]`` containes the dirichlet value of dof ``dofs[i]``, or b) the length of all dofs of the function space, i.e. ``u[dofs[i]]`` containes the dirichlet value of dof ``dofs[i]``. Additionally, for type conversion, ``u`` can also be a :class:`dolfinx.fem.DirichletBC`, in which case the corresponding dofs and values are extracted from it. (Because :class:`dolfinx.fem.DirichletBC` only stores the Cpp object of its function space, the function space must be provided again in this case.)
            dofs:
                The dofs of the function space on which the Dirichlet condition is imposed. This argument is mandatory unless ``u`` is a :class:`dolfinx.fem.DirichletBC`.
                
        .. note:: 
            This class does not really contain new functionality / information compared to :class:`dolfinx.fem.DirichletBC` and was introduced for making this addon more convenient to code.
        """
        
        if isinstance(u, fem.DirichletBC):
            if dofs is not None:
                raise ValueError('dofs must not be provided if u is a dolfinx.fem.DirichletBC')
            if not u.function_space == function_space._cpp_object:
                raise ValueError('Function space must be the same as in the provided dolfinx.fem.DirichletBC')
            dofs = u.dof_indices()[0]
            u = u.g
        
        if isinstance(u, fem.Function) or (type(u).__name__.startswith('Function_') and type(u).__module__.startswith('dolfinx')):
            if not (u.function_space == function_space or u.function_space == function_space._cpp_object):
                raise ValueError('Function u must be defined on the same function space as the DirichletBC')
            values = u.x.array[dofs]
        elif isinstance(u, fem.Constant) or (type(u).__name__.startswith('Constant_') and type(u).__module__.startswith('dolfinx')):
            values = np.full(dofs.shape, u.value, dtype=default_scalar_type)
        elif isinstance(u, Number):
            values = np.full(dofs.shape, default_scalar_type(u))
        else:
            try:
                values = as_numpy_vector(u, default_scalar_type)
            except:
                try:
                    u_func = fem.Function(function_space)
                    u_func.interpolate(u)
                    values = u_func.x.array[dofs]
                except:
                    raise TypeError('dont know what to do')
        
        dofs   = as_numpy_vector(dofs, np.int32)
        values = as_numpy_vector(values, default_scalar_type)
        
        def unique(x): seen = set(); return not any(i in seen or seen.add(i) for i in x)
        if not unique(dofs):
            raise ValueError('duplicate dofs in DirichletBC')
        
        if len(values) == function_space.dofmap.index_map.size_global:
            values = values[dofs]
        elif len(values) != len(dofs):
            raise ValueError('Number of values does not match the number of dirichlet dofs or function space dofs')
        
        idx = np.argsort(dofs)
        
        self.fixed_dofs: np.ndarray[int] = dofs[idx]
        """Fixed dofs corresponding to the dirichlet values (unique, i.e. no duplicates, and sorted in ascending order)."""
        self.values: np.ndarray[float] = values[idx]
        """Discrete dirichlet values, i.e. ``values[i]`` is the dirichlet value at dof ``fixed_dofs[i]``."""
        self.function_space: FunctionSpace = function_space
        """Function space the boundary condition is defined on."""
        self._free_dofs = None
        self.ndofs: int = self.function_space.dofmap.index_map.size_global
        """Total number of dofs of the function space (free+fixed)."""
        
    @property
    def free_dofs(self) -> np.ndarray[int]:
        """Free dofs where no dirichlet condition is applied (unique, i.e. no duplicates, and sorted in ascending order)."""
        if self._free_dofs is None:
            free_dofs = np.ones(self.ndofs, dtype=bool)
            free_dofs[self.fixed_dofs] = False
            self._free_dofs = as_numpy_vector(np.sort(np.where(free_dofs)[0]), np.int32)
        return self._free_dofs
    
    def to_dolfinx(self) -> fem.DirichletBC:
        """Convert this Dirichlet boundary condition to a :class:`dolfinx.fem.DirichletBC`."""
        u = fem.Function(self.function_space)
        u.x.array[:] = default_scalar_type(0)
        u.x.array[self.fixed_dofs] = self.values
        return fem.dirichletbc(u, self.fixed_dofs)

      
def dirichletbc(function_space: FunctionSpace, u: BoundaryFunctionType, facets: np.ndarray[int] | None = None) -> DirichletBC:
    r"""Create a Dirichlet boundary condition for ``function_space`` on the boundary part ``facets`` with the boundary values given by ``u``.
    
    This function operates on **facets not dofs**, unlike :func:`dolfinx.fem.dirichletbc`. This decision was made conserning Petrov-Galerkin formulations: the dofs of the test and trial space may differ, while the facets belong to the mesh and are thus independend of the space (thus avoiding potential hard to debug errors by defining dirichlet condition for one space while using the facet dofs generated based on another space). 
    If one wants to create a dirichlet boundary condition based on dofs, one can always use :class:`DirichletBC` directly. 
    
    Args:
        function_space:
            The function space the boundary condition is defined on.
        u:
            The boundary value. Can be a number, :class:`dolfinx.fem.Constant`, :class:`dolfinx.fem.Function` or any object that can be interpolated into a :class:`dolfinx.fem.Function`.
        facets:
            The boundary facets where the Dirichlet condition is applied. If ``None``, all exterior facets are used.
    """
   
    if facets is None:
        facets = mesh.exterior_facet_indices(function_space.mesh.topology)
    
    facets = as_numpy_vector(facets, np.int32)
    bdry_dofs = fem.locate_dofs_topological(function_space, function_space.mesh.topology.dim-1, facets)
    
    return DirichletBC(function_space, u, bdry_dofs)




def collect_dirichletbcs(bcs: Iterable[DirichletBC | fem.DirichletBC] | None, function_space: FunctionSpace | Iterable[FunctionSpace] | None = None, check_tol: float = 1e-14) -> DirichletBC | list[DirichletBC]:
    r"""Collect multiple Dirichlet boundary conditions given in ``bcs`` into a single Dirichlet boundary condition for each space given in ``function_space``.
    
    Args:
        bcs:
            Iterable of Dirichlet boundary conditions (possibly defined on different function spaces) to be collected into one Dirichlet boundary condition per function space. If bcs contains any :class:`dolfinx.fem.DirichletBC`, the `function_space` argument is mandatory. If bcs is None, an empty Dirichlet boundary condition is returned for each function space.
            
        function_space:
            The function space(s) for which the Dirichlet boundary conditions are to be collected. If ``None``, all function spaces occuring in ``bcs`` are used, in the same order as they appear in ``bcs``. If there is no Dirichlet boundary condition for a function space, an empty Dirichlet boundary condition is returned for that space.
            
        check_tol:  
            Tolerance for checking conflicting Dirichlet values at the same dof. If two Dirichlet boundary conditions specify different values at the same dof, a warning is issued and the value from the first Dirichlet boundary condition in the ``bcs`` is used. This option as no impact on the output and is just to inform about conflicting conditions. To turn off this check, set ``check_tol`` to ``np.inf``.
        
    Returns:
        One Dirichlet boundary condition per function space containing all Dirichlet boundary conditions for that space. If ``function_space`` is a single function space, a single Dirichlet boundary condition is returned; if it is an iterable of function spaces or ``None``, a list of Dirichlet boundary conditions is returned in the same order as the function spaces in ``function_space``.
    """
    
    if bcs is None and function_space is None:
        raise ValueError('bcs and function_space can not both be None at the same time')
    
    if function_space is None:
        function_space = list()
        for bc in bcs:
            space = bc.function_space
            if space not in function_space:
                function_space.append(space)
    
    # if more than one function space, return one BC for each space  
    if isinstance(function_space, Iterable):
        return [collect_dirichletbcs(bcs, fs, check_tol) for fs in function_space]
    
    if bcs is None: # empty bc
        return DirichletBC(function_space, np.array([],dtype=default_scalar_type), np.array([],dtype=np.int32))
    
    # convert dolfinx DirichletBCs to pgfenicsx DirichletBCs
    bcs_converted = []
    for bc in bcs:
        if isinstance(bc, fem.DirichletBC):
            if function_space is None:
                raise ValueError('function_space must be provided if bcs contains dolfinx.fem.DirichletBCs')
            if bc.function_space == function_space._cpp_object:
                bcs_converted.append(DirichletBC(function_space, bc))
        elif bc.function_space == function_space:
            bcs_converted.append(bc)
    bcs = bcs_converted
    
    if len(bcs) == 0:      # empty bc
        return DirichletBC(function_space, np.array([],dtype=default_scalar_type), np.array([],dtype=np.int32))
    elif len(bcs) == 1:    # only one bc, nothing to collect
        return bcs[0]
    
    all_dofs = np.empty((1, 0), np.int32)
    all_values = np.empty((1, 0), default_scalar_type)
    
    for bc in bcs:
        dofs = bc.fixed_dofs
        values = bc.values
        
        idx_duplicates = np.isin(dofs, all_dofs, assume_unique=True)
        idx_new = np.logical_not(idx_duplicates)
        all_dofs = np.append(all_dofs, dofs[idx_new])
        all_values = np.append(all_values, values[idx_new])
        
        for dupl_dof, dupl_value in zip(dofs[idx_duplicates], values[idx_duplicates]):
            existing_value = all_values[np.where(all_dofs == dupl_dof)][0]
            if not np.isclose(existing_value, dupl_value, atol=check_tol):
                from warnings import warn
                warn(f'\nConflicting Dirichlet BC value at dof#{dupl_dof}: {dupl_value} vs {existing_value}; using {existing_value}')
                
    return DirichletBC(function_space, all_values, all_dofs)


def assemble_system(F: ufl.form.Form | tuple[ufl.form.Form, ufl.form.Form], bcs: Iterable[DirichletBC] | None = None, petsc: bool = False) -> tuple[sparse.csr_array, np.ndarray] | tuple[PETSc.Mat, PETSc.Vec]:
    r"""
    Assemble the system matrix and right-hand side of a Petrov-Galerkin variational problem either as SciPy sparse matrix or using PETSc, with the Dirichlet boundary conditions given in ``bcs``.
    
    Args:
        F:
            The variational form, either one form that can be split using ``A,l = ufl.system(F)`` or a tuple ``(A,l)`` of a bilinear form ``A`` and a linear form ``l``.
        bcs:
            Iterable of Dirichlet boundary conditions to be applied, can contain different Dirichlet boundary conditions for the trial and test space.
        petsc:
            If True, assemble the system using PETSc matrices and vectors, otherwise using scipy and numpy.
    Returns: (A,l)
        The assembled system matrix ``A`` and right-hand side ``l``.
        
    Example:
        >>> from dolfinx import fem, mesh
        >>> import pgfenicsx
        >>> # ... mesh creation ...
        >>> U = fem.functionspace(...)
        >>> V = fem.functionspace(...)
        >>> # ... boundary facets location ...
        >>> bcs = [pgfenicsx.dirichletbc(U, uD, facets_U), 
                   pgfenicsx.dirichletbc(V, vD, facets_V)]
        >>> # ... define variational forms A and l ...
        >>> # solving the system:
        >>> # (using QR as direct solver for PG formulations
        >>> # as there might occure non-quadratic matrices)
        
        >>> # Using scipy:
        >>> from scipy.sparse.linalg import lsqr
        >>> A_,l_ = pgfenicsx.assemble_system((A,l), bcs, petsc=False)
        >>> u = fem.Function(U)
        >>> u.x.array[:] = lsqr(A_,l_)[0] 
        
        >>> # Using PETSc:
        >>> from petsc4py import PETSc
        >>> A_,l_ = pgfenicsx.assemble_system((A,l), bcs, petsc=True)
        >>> solver = PETSc.KSP().create(MPI.COMM_WORLD)
        >>> solver.setOperators(A_)
        >>> solver.setType("preonly")
        >>> solver.getPC().setType("qr")
        >>> solver.setFromOptions()
        >>> u = fem.Function(U)
        >>> u.x.petsc_vec.ghostUpdate(
                addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
        >>> )
        >>> solver.solve(l_, u.x.petsc_vec)
        
    .. note::
        If you call ``assemble_system`` for the same list of Dirichlet boundary conditions ``bcs`` multiple times, you might run ``bcs = pgfenicsx.collect_dirichletbcs(bcs, [trial_space, test_space])`` beforehand to avoid collecting them in every call of ``assemble_system``. This has no impact on functionality, but might improve performance.
        
    """
    if isinstance(F, tuple):
        A,l = F
    else:
        A,l = ufl.system(F)
    trial_space = A.arguments()[1].ufl_function_space()
    test_space  = A.arguments()[0].ufl_function_space()
    
    if not test_space == l.arguments()[0].ufl_function_space():
        raise ValueError("Test space of A and l do not match.")
    
    [trial_bc,test_bc] = collect_dirichletbcs(bcs, [trial_space, test_space])
    
    if petsc:
        A = _assemble_matrix_PETSc(A, trial_bc, test_bc)
        l = _assemble_vector_PETSc(l, trial_bc, test_bc)
    else:
        A = _assemble_matrix_scipy(A, trial_bc, test_bc)
        l = _assemble_vector_scipy(l, trial_bc, test_bc)
    return A,l

def assemble_matrix(A: ufl.form.Form, bcs: Iterable[DirichletBC] | None = None, petsc: bool = False) -> sparse.csr_array | PETSc.Mat:
    
    trial_space = A.arguments()[1].ufl_function_space()
    test_space  = A.arguments()[0].ufl_function_space()
    
    [trial_bc,test_bc] = collect_dirichletbcs(bcs, [trial_space, test_space])
    
    if petsc:
        A = _assemble_matrix_PETSc(A, trial_bc, test_bc)
    else:
        A = _assemble_matrix_scipy(A, trial_bc, test_bc)
    return A
    

def assemble_vector(l: ufl.form.Form, trial_space, bcs: Iterable[DirichletBC] | None = None, petsc: bool = False) -> np.ndarray | PETSc.Vec:
    
    test_space = l.arguments()[0].ufl_function_space()
    
    [trial_bc,test_bc] = collect_dirichletbcs(bcs, [trial_space, test_space])
    
    if petsc:
        l = _assemble_vector_PETSc(l, trial_bc, test_bc)
    else:
        l = _assemble_vector_scipy(l, trial_bc, test_bc)
    return l
    
    
def _assemble_matrix_scipy(A: ufl.form.Form, trial_bc: DirichletBC, test_bc: DirichletBC) -> sparse.csr_array:
    A = fem.assemble_matrix(fem.form(A)).to_scipy()
    if len(trial_bc.fixed_dofs) == 0 and len(test_bc.fixed_dofs) == 0:
        return A
    n_diri = len(trial_bc.fixed_dofs)
    A_dirichlet = sparse.coo_array((np.ones(n_diri), (np.arange(n_diri), trial_bc.fixed_dofs)), shape=(n_diri, A.shape[1])).tocsr()
    A = sparse.vstack([A_dirichlet, A[test_bc.free_dofs,:]])
    return A
    # A = A[test_bc.free_dofs,:].tocsc()
    # l = l[test_bc.free_dofs] - A[:, trial_bc.fixed_dofs] @ trial_bc.values
    # A = A[:, trial_bc.free_dofs]
    # return A,l,trial_bc

def _assemble_vector_scipy(l: ufl.form.Form, trial_bc: DirichletBC, test_bc: DirichletBC) -> np.ndarray:
    l = fem.assemble_vector(fem.form(l)).array
    if len(trial_bc.fixed_dofs) == 0 and len(test_bc.fixed_dofs) == 0:
        return l    
    return np.concatenate((trial_bc.values, l[test_bc.free_dofs]))


def _assemble_matrix_PETSc(A: ufl.form.Form, trial_bc: DirichletBC, test_bc: DirichletBC) -> PETSc.Mat:
    n = trial_bc.ndofs

    n_diri = len(trial_bc.fixed_dofs)   # number of rows added to set the trial dirichlet values
    
    A = dolfinx.fem.petsc.assemble_matrix(fem.form(A))
    A.assemble()
    
    if len(trial_bc.fixed_dofs) == 0 and len(test_bc.fixed_dofs) == 0:
        return A
    
    A_format = A.getType()
    A_comm = A.comm
    
    # ToDo: if len(trial_bc.fixed_dofs) == 0: just remove test dirichlet rows
    # ToDO: if len(test_bc.fixed_dofs) == 0: just set trial dirichlet rows
    # ToDo: if len(trial_bc.fixed_dofs) == len(test_bc.fixed_dofs): dont create a new matrix, modify A in place

    # Extract part not deleted by test dirichlet BC
    is_rows = PETSc.IS().createGeneral(test_bc.free_dofs, comm=A_comm)
    is_col = PETSc.IS().createGeneral(np.arange(n, dtype=np.int32), comm=A_comm)
    A_free = A.createSubMatrix(is_rows, is_col)
    A.destroy()
    is_rows.destroy()
    is_col.destroy()
    
    # Create new matrix to enforce trial dirichlet BC
    A_dirichlet = PETSc.Mat().createAIJ(size=(n_diri, n), nnz=1, comm=A_comm)
    A_dirichlet.setUp()
    for i, j in enumerate(trial_bc.fixed_dofs):
        A_dirichlet[i, j] = 1.0
    
    # merge both matrices
    A_ = PETSc.Mat().createNest([[A_dirichlet],[A_free]],comm=A_comm)
    A_dirichlet.destroy()
    A_free.destroy()

    A_.assemble()
    A_.convert(A_format)
    return A_

def _assemble_vector_PETSc(l: ufl.form.Form, trial_bc: DirichletBC, test_bc: DirichletBC) -> PETSc.Vec:
   
    n_diri = len(trial_bc.fixed_dofs)   # number of rows added to set the trial dirichlet values
    n_free = len(test_bc.free_dofs)     # number of rows remaining afte removing the test dirichlet rows 
    
    l = dolfinx.fem.petsc.assemble_vector(fem.form(l))
    l.assemble()
    
    if len(trial_bc.fixed_dofs) == 0 and len(test_bc.fixed_dofs) == 0:
        return l
    
    # ToDo: if len(trial_bc.fixed_dofs) == 0: just remove test dirichlet rows
    # ToDO: if len(test_bc.fixed_dofs) == 0: just set trial dirichlet rows
    # ToDo: if len(trial_bc.fixed_dofs) == len(test_bc.fixed_dofs): dont create a new matrix, modify A in place
    
    # setup right hand side
    l_ = PETSc.Vec().create(comm=l.comm)
    l_.setType(l.getType())
    l_.setSizes(n_diri+n_free)
    l_.setUp()
    
    # Extract part not deleted by test dirichlet BC
    is_rows = PETSc.IS().createGeneral(test_bc.free_dofs, comm=l.comm)
    l_dirichlet = l.getSubVector(is_rows)        
    l_.setValues(range(n_diri, n_diri + n_free), l_dirichlet.getArray())
    l.restoreSubVector(is_rows, l_dirichlet)
    l_dirichlet.destroy()
    l.destroy()
    is_rows.destroy()
    # Set part to to enforce trial dirichlet BC
    l_.setValues(range(n_diri), trial_bc.values)
    l_.assemble()
    return l_