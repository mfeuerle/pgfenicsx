# Moritz Feuerle, 2025

__all__ = ['merge_dirichletbcs', 'apply_dirichletbc_system','apply_dirichletbc_vector', 'apply_dirichletbc_matrix']


import numpy as np
import scipy.sparse as sparse
from dolfinx import fem, cpp
from petsc4py import PETSc

from typing import TypeAlias
from collections.abc import Iterable
DirichletCPPType: TypeAlias = cpp.fem.DirichletBC_float32 | cpp.fem.DirichletBC_float64 | cpp.fem.DirichletBC_complex64 | cpp.fem.DirichletBC_complex128
FunctionSpaceCPPType: TypeAlias = cpp.fem.FunctionSpace_float32 | cpp.fem.FunctionSpace_float64


def _as_numpy_vector(x, dtype=None):
    """Ensure that x is a numpy array of the given dtype and shape (n,)."""
    return np.asarray(x, dtype=dtype).reshape((-1,))
    
    
def merge_dirichletbcs(bcs: Iterable[fem.DirichletBC | DirichletCPPType] | None, function_space: fem.FunctionSpace | FunctionSpaceCPPType | Iterable[fem.FunctionSpace | FunctionSpaceCPPType] | None = None, check_tol: float = 1e-14) -> DirichletCPPType | list[DirichletCPPType]:
    r"""Merge a list of Dirichlet boundary conditions into a single Dirichlet boundary condition for each space.
    
    Args:
        bcs:
            Iterable of Dirichlet boundary conditions (possibly defined on different function spaces) to be collected into one Dirichlet boundary condition per function space. If bcs is ``None``, an empty Dirichlet boundary condition is returned for each function space.
            
        function_space:
            The function space(s) for which the Dirichlet boundary conditions are to be collected. If ``None``, all function spaces occuring in ``bcs`` are used, in the same order as they appear in ``bcs``. If there is no Dirichlet boundary condition for a function space, an empty Dirichlet boundary condition is returned for that space.
            
        check_tol:
            Tolerance for checking conflicting Dirichlet values at the same dof. If two Dirichlet boundary conditions specify different values at the same dof, a warning is issued and the value from the first Dirichlet boundary condition in the list ``bcs`` is used. This option has no impact on the output and is just to inform about conflicting Dirichlet conditions. To turn this check off, set ``check_tol=np.inf``.
            
    Returns:
        One Dirichlet boundary condition per function space in which all Dirichlet boundary conditions for that space are merged. If ``function_space`` is a single function space, a single Dirichlet boundary condition is returned; if it is an iterable of function spaces or ``None``, a list of Dirichlet boundary conditions is returned in the same order as the function spaces in ``function_space``.
        
    .. note::
        This method is called inside :func:`apply_dirichletbc_vector`, :func:`apply_dirichletbc_matrix` and :func:`apply_dirichletbc_system` to merge multiple Dirichlet boundary conditions before applying them to the system, thus it is not necessary to call this method explicitly. However, to avoid merging the same Dirichlet boundary conditions multiple times, or to turn off the warnings using ``check_tol``, it can be applyed manually by ``bcs = merge_dirichletbcs(bcs[, check_tol])`` before calling any of the above functions.
    """
    
    if bcs is None and function_space is None:
        raise ValueError('bcs and function_space can not both be None at the same time')
    
    # get all function spaces if not provided
    if function_space is None:
        function_space = list()
        for bc in bcs:
            space = bc.function_space
            if space not in function_space:
                function_space.append(space)
                
    # convert all function space to cpp objects
    if isinstance(function_space, fem.FunctionSpace):
        function_space = function_space._cpp_object
        
    # if more than one function space, return one BC for each space                  
    if isinstance(function_space, Iterable):
        return [merge_dirichletbcs(bcs, fs, check_tol) for fs in function_space]
    # from now on, function_space is a single space
    
    if bcs is not None:
        # convert bcs to cpp objects
        bcs = [bc._cpp_object if isinstance(bc, fem.DirichletBC) else bc for bc in bcs]

        # get all dofs and values    
        dofs = []
        values = []
        for bc in bcs:
            if bc.function_space != function_space:
                continue
            d = np.unique(bc.dof_indices()[0])
            if type(bc.value).__name__.startswith('Function_'):
                v = bc.value.x.array[d]
            elif type(bc.value).__name__.startswith('Constant_'):
                v = np.full(d.shape, bc.value.value, dtype=bc.value.value.dtype)
            else:
                raise NotImplementedError(f'Dont know the DirichletBC value type {type(bc.value)}')
            dofs.append(d)
            values.append(v)
            
        if len(dofs) == 0:
            # no dirichlet bcs for this function space
            bcs = None
        else:
            # collect dofs and values in one array, checking for conflicts
            dtype = np.common_type(function_space.mesh.geometry.x, *values)
            all_dofs = _as_numpy_vector(dofs[0],dtype=np.int32)
            all_values = _as_numpy_vector(values[0],dtype=dtype)
            for d,v in zip(dofs[1:], values[1:]):
                idx_duplicates = np.isin(d, all_dofs, assume_unique=True)
                idx_new = np.logical_not(idx_duplicates)
                all_dofs = np.append(all_dofs, d[idx_new])
                all_values = np.append(all_values, _as_numpy_vector(v[idx_new], dtype=dtype))
                
                for d_dupl, v_dupl in zip(d[idx_duplicates], _as_numpy_vector(v[idx_duplicates], dtype=dtype)):
                    v_existing = all_values[np.where(all_dofs == d_dupl)][0]
                    if not np.isclose(v_existing, v_dupl, atol=check_tol):
                        from warnings import warn
                        warn(f'\nConflicting Dirichlet BC value at dof#{d_dupl}: {v_dupl} vs {v_existing}; using {v_existing}')
                    
    if bcs is None:
        dtype = function_space.mesh.geometry.x.dtype
        all_dofs = np.array([], dtype=np.int32)
        all_values = np.array([], dtype=dtype)
    
    # select right cpp class based on dtype           
    if np.issubdtype(dtype, np.float32):
        bctype = cpp.fem.DirichletBC_float32
        fntype = cpp.fem.Function_float32
    elif np.issubdtype(dtype, np.float64):
        bctype = cpp.fem.DirichletBC_float64
        fntype = cpp.fem.Function_float64
    elif np.issubdtype(dtype, np.complex64):
        bctype = cpp.fem.DirichletBC_complex64
        fntype = cpp.fem.Function_complex64
    elif np.issubdtype(dtype, np.complex128):
        bctype = cpp.fem.DirichletBC_complex128
        fntype = cpp.fem.Function_complex128
    else:
        raise NotImplementedError(f"Type {dtype} not supported.")
    
    # build the new DirichletBC
    u = fntype(function_space)
    u.x.array[:] = 0.0
    u.x.array[all_dofs] = all_values
    return bctype(u, all_dofs)


def apply_dirichletbc_vector(l: np.ndarray | PETSc.Vec, trial: fem.FunctionSpace, test: fem.FunctionSpace, bcs: fem.DirichletBC) -> np.ndarray | PETSc.Vec:
    r"""Apply the Dirichlet boundary conditions ``bcs`` to the right-hand side vector ``l`` of a Petrov-Galerkin variational problem.
    
    See :meth:`apply_dirichletbc_system` for a documentation of how the Dirichlet boundary conditions are applied to the right-hand side vector.
    
    Args:
        l:
            The right-hand side vector to which the Dirichlet boundary conditions are to be applied.
        trial:
            The trial function space associated with the vector ``l``.
        test:
            The test function space associated with the vector ``l``.
        bcs:
            The Dirichlet boundary conditions to be applied.
            
    Returns:
        The right-hand side vector ``l`` with the Dirichlet boundary conditions applied.
        
    .. note::
       It might be advisable to call :func:`merge_dirichletbcs` before this function.
    """

    return _apply_dirichletbc_vector(l,*_extract_dirichletbcs(trial, test, bcs))

def apply_dirichletbc_matrix(A: sparse.csr_array | PETSc.Mat, trial: fem.FunctionSpace, test: fem.FunctionSpace, bcs: fem.DirichletBC) -> sparse.csr_array | PETSc.Mat:
    r"""Apply the Dirichlet boundary conditions ``bcs`` to the system matrix ``A`` of a Petrov-Galerkin variational problem.
    
    See :meth:`apply_dirichletbc_system` for a documentation of how the Dirichlet boundary conditions are applied to the system matrix.
    
    Args:
        A:
            The system matrix to which the Dirichlet boundary conditions are to be applied.
        trial:
            The trial function space associated with the matrix ``A``.
        test:
            The test function space associated with the matrix ``A``.
        bcs:
            The Dirichlet boundary conditions to be applied.
            
    Returns:
        The system matrix ``A`` with the Dirichlet boundary conditions applied.
        
    .. note::
       It might be advisable to call :func:`merge_dirichletbcs` before this function.
    """
    
    return _apply_dirichletbc_matrix(A,*_extract_dirichletbcs(trial, test, bcs))

def apply_dirichletbc_system(A: sparse.csr_array | PETSc.Mat, l: np.ndarray | PETSc.Vec, trial: fem.FunctionSpace, test: fem.FunctionSpace, bcs: fem.DirichletBC) -> tuple[sparse.csr_array | PETSc.Mat, np.ndarray | PETSc.Vec]:
    r"""Apply the Dirichlet boundary conditions ``bcs`` to the system matrix ``A`` and right-hand side vector ``l`` of a Petrov-Galerkin variational problem.
    
    Let :math:`U` be the trial and :math:`V` the test function space, and denote by :math:`d_U` and :math:`d_V` the sets of dofs in :math:`U` and :math:`V` restricted by the Dirichlet boundary conditions in ``bcs``, while :math:`f_U` and :math:`f_V` denote the set of dofs in :math:`U` and :math:`V` not restricted by the Dirichlet boundary conditions in ``bcs``. Further, :math:`g` denotes the Dirichlet function values of the trial space.
    
    The initial system :math:`A u = l` can be partitioned into the following parts:
    
    .. math::
        A = \begin{bmatrix} A_{f_V,f_U} & A_{f_V,d_U} \\ A_{d_V,f_U} & A_{d_V,d_U}  \end{bmatrix}, 
        \quad u = \begin{bmatrix} u_{f_U} \\ u_{d_U} \end{bmatrix},
        \quad l = \begin{bmatrix} l_{f_V} \\ l_{d_V} \end{bmatrix}.
        
    Now, we apply the dirichlet BCs to this system. First, we remove the rows corresponding to the test dirichlet dofs :math:`d_V`, i.e.
    
    .. math::
        \begin{bmatrix} A_{f_V,f_U} & A_{f_V,d_U} \end{bmatrix} 
        \begin{bmatrix} u_{f_U} \\ u_{d_U} \end{bmatrix} 
        = l_{f_V}.
        
    To enforce the dirichlet values on the trial dofs :math:`d_U`, we add a new set of rows to the system:
    
    .. math::
        \tilde{A} := \begin{bmatrix} A_{f_V,f_U} & A_{f_V,d_U} \\ 0 & I_{d_U,d_U} \end{bmatrix}
        \begin{bmatrix} u_{f_U} \\ u_{d_U} \end{bmatrix}
        = \begin{bmatrix} l_{f_V} \\ g \end{bmatrix} =: \tilde{l}.
    
    Thus, the system now reads :math:`\tilde{A} u = \tilde{l}`. The resulting system matrix is in general neither square nor symmetric, even if the original system matrix was square and symmetric. This is inherent to Petrov-Galerkin formulations where the trial and test spaces can differ, thus one can not expect the system to be square or even symmetric.
    
    See also :meth:`apply_dirichletbc_matrix` and :meth:`apply_dirichletbc_vector`.
    
    Args:
        A:
            The system matrix to which the Dirichlet boundary conditions are to be applied.
        l:
            The right-hand side vector to which the Dirichlet boundary conditions are to be applied.
        trial:
            The trial function space associated with the matrix ``A`` and vector ``l``.
        test:
            The test function space associated with the matrix ``A`` and vector ``l``.
        bcs:
            The Dirichlet boundary conditions to be applied.
            
    Returns:
        A tuple ``(A, l)`` containing the system matrix and right-hand side vector with the Dirichlet boundary conditions applied.
        
    .. note::
       It might be advisable to call :func:`merge_dirichletbcs` before this function.        
    """
    data = _extract_dirichletbcs(trial, test, bcs)
    A = _apply_dirichletbc_matrix(A, *data)
    l = _apply_dirichletbc_vector(l, *data)
    return A, l


def _extract_dirichletbcs(trial: fem.FunctionSpace, test: fem.FunctionSpace, bcs: fem.DirichletBC):
    r"""Extract the necessary data from the Dirichlet boundary conditions to apply them to the system matrix and vector.
    Args:
        trial:
            The trial function space.
        test:
            The test function space.
        bcs:
            The Dirichlet boundary conditions to be applied.
    Returns: trial_fixed, test_fixed, test_free, values, n, m
        trial_fixed: Dofs in the trial space restricted by Dirichlet BCs.
        test_fixed: Dofs in the test space restricted by Dirichlet BCs.
        test_free: Dofs in the test space not restricted by Dirichlet BCs.
        values: Values of the trial BC at the fixed dofs.
        n: Total number of dofs in the trial space.
        m: Total number of dofs in the test space.
    """
    
    if trial == test:
        from warnings import warn
        warn('Trial and test space are equal; consider using standard FEniCSx methods.')
    
    [trial_bc,test_bc] = merge_dirichletbcs(bcs, [trial, test])
    
    # total number of dofs / matrix dimensions
    n = trial_bc.function_space.dofmap.index_map.size_global
    m = test_bc.function_space.dofmap.index_map.size_global
    # dofs restricted by dirichlet BCs
    trial_fixed = trial_bc.dof_indices()[0]
    test_fixed  = test_bc.dof_indices()[0]
    # free dofs not resticted by dirichlet BCs
    test_free = np.arange(m,dtype=np.int32); test_free = np.delete(test_free, test_fixed)
    # values of the trial bc at the fixed dofs
    values = trial_bc.value.x.array[trial_fixed]
    
    return trial_fixed, test_fixed, test_free, values, n, m
    
    
    
def _apply_dirichletbc_vector(l, trial_fixed, test_fixed, test_free, values, n, m):
    
    if len(test_fixed) == 0 and len(trial_fixed) == 0:
        return l
    # ToDo: if len(trial_bc.fixed_dofs) == 0: just remove test dirichlet rows
    # ToDO: if len(test_bc.fixed_dofs) == 0: just set trial dirichlet rows
    # ToDo: if len(trial_bc.fixed_dofs) == len(test_bc.fixed_dofs): dont create a new matrix, modify A in place

    if isinstance(l, np.ndarray):
        return np.concatenate((values, l[test_free]))
    
    elif isinstance(l, PETSc.Vec):
        l.assemble()  
        # create right hand side
        l_ = PETSc.Vec().create(comm=l.comm)
        l_.setType(l.getType())
        l_.setSizes(len(trial_fixed)+len(test_free))
        l_.setUp()
        
        # set trial dirichlet values
        l_.setValues(range(len(trial_fixed)), values)
        
        # insert values that are not test dirichlet rows
        is_rows = PETSc.IS().createGeneral(test_free, comm=l.comm)
        l_diri = l.getSubVector(is_rows)
        l_.setValues(range(len(trial_fixed), len(trial_fixed) + len(test_free)), l_diri.getArray())
        l.restoreSubVector(is_rows, l_diri)
        l.destroy()
        is_rows.destroy()
        
        l_.assemble()
        return l_
    
    else: 
        raise TypeError(f'Unknown vector type {type(l)}')
    
    
def _apply_dirichletbc_matrix(A, trial_fixed, test_fixed, test_free, values, n, m):
    
    if len(test_fixed) == 0 and len(trial_fixed) == 0:
        return A
    
    # ToDo: if len(trial_bc.fixed_dofs) == 0: just remove test dirichlet rows
    # ToDO: if len(test_bc.fixed_dofs) == 0: just set trial dirichlet rows
    # ToDo: if len(trial_bc.fixed_dofs) == len(test_bc.fixed_dofs): dont create a new matrix, modify A in place
    
    if sparse.issparse(A):
        # rows to enforce the trial dirichlet values
        A_diri = sparse.coo_array((np.ones(len(trial_fixed)), (np.arange(len(trial_fixed)), trial_fixed)), shape=(len(trial_fixed), n)).tocsr()
        # add trial dirichlet rows and remove test dirichlet rows
        return sparse.vstack([A_diri, A[test_free,:]])
    
    elif isinstance(A, PETSc.Mat):
        A.assemble()
        A_type = A.getType()
        A_comm = A.comm
    
         # remove test dirichlet rows
        is_rows = PETSc.IS().createGeneral(test_free, comm=A_comm)
        is_col  = PETSc.IS().createGeneral(np.arange(n, dtype=np.int32), comm=A_comm)
        A_free = A.createSubMatrix(is_rows, is_col)
        A.destroy()
        is_rows.destroy()
        is_col.destroy()
        
        # rows to enforce the trial dirichlet values
        A_diri = PETSc.Mat().createAIJ(size=(len(trial_fixed), n), nnz=1, comm=A_comm)
        A_diri.setUp()
        for i, j in enumerate(trial_fixed): A_diri[i, j] = 1.0
        
        # merge both matrices
        A = PETSc.Mat().createNest([[A_diri],[A_free]], comm=A_comm)
        A_diri.destroy()
        A_free.destroy()
        
        # restore original matrix type
        A.assemble()
        A.convert(A_type).assemble()
        return A
    
    else: 
        raise TypeError(f'Unknown matrix type {type(A)}')




# def assemble_matrix(A: fem.Form, bcs: Iterable[fem.DirichletBC | DirichletCPPType] | None = None, petsc: bool = False, **kwargs) -> sparse.csr_array | PETSc.Mat:
#     r"""Assemble the system matrix of a Galerkin or Petrov-Galerkin variational problem either as SciPy sparse matrix or PETSc matrix, restricted by the Dirichlet boundary conditions given in ``bcs``.
    
#     For Petrov-Galerkin problems, the trial and test spaces can be different, thus ``bcs`` can contain different Dirichlet boundary conditions for the trial and test space. The Dirichlet boundary conditions on the test space are always assumed to be homogeneous (zero), independent of the values specified in the Dirichlet boundary conditions related to the test space. Thus, it also supports standard Galerkin problems as a special case where the trial and test space are the same.
    
#     Args:
#         A:
#             The bilinear form to be assembled.
#         bcs:
#             Iterable of Dirichlet boundary conditions to be applied, can contain different Dirichlet boundary conditions for the trial and test space.
#         petsc:
#             If True, assemble the matrix as PETSc matrix, otherwise as scipy sparse matrix.
#         kwargs:
#             Additional keyword arguments passed to either :func:`dolfinx.fem.assemble_matrix` or :func:`dolfinx.fem.petsc.assemble_matrix`, depending on the value of ``petsc``.
#     Returns:
#         The assembled system matrix ``A``.
        
#     .. note::
#         It might be advisable to call :meth:`merge_dirichletbcs` before this function.
#     """

    
#     trial_space = A.function_spaces[1]
#     test_space  = A.function_spaces[0]
    
#     [trial_bc,test_bc] = merge_dirichletbcs(bcs, [trial_space, test_space])
    
#     if petsc:
#         A = _assemble_matrix_PETSc(A, trial_bc, test_bc, **kwargs)
#     else:
#         A = _assemble_matrix_scipy(A, trial_bc, test_bc, **kwargs)
#     return A
    

# def assemble_vector(l: fem.Form, trial_space: fem.FunctionSpace | FunctionSpaceCPPType, bcs: Iterable[fem.DirichletBC | DirichletCPPType] | None = None, petsc: bool = False, **kwargs) -> np.ndarray | PETSc.Vec:
#     r"""Assemble the tright-hand side vector of a Galerkin or Petrov-Galerkin variational problem either as Numpy array or PETSc vector, restricted by the Dirichlet boundary conditions given in ``bcs``.
    
#     For Petrov-Galerkin problems, the trial and test spaces can be different, thus ``bcs`` can contain different Dirichlet boundary conditions for the trial and test space. The Dirichlet boundary conditions on the test space are always assumed to be homogeneous (zero), independent of the values specified in the Dirichlet boundary conditions related to the test space. Thus, it also supports standard Galerkin problems as a special case where the trial and test space are the same.
    
#     As the right hand side ``l`` is defined purely on the test space, its associated trial space is ambigous, and thus has to be passed explicitly. 
    
#     Args:
#         l:
#             The linear form to be assembled.
#         trial_space:
#             The trial function space associated with the linear form.
#         bcs:
#             Iterable of Dirichlet boundary conditions to be applied, can contain different Dirichlet boundary conditions for the trial and test space.
#         petsc:
#             If True, assemble the vector as PETSc vector, otherwise as numpy array.
#         kwargs:
#             Additional keyword arguments passed to either :func:`dolfinx.fem.assemble_vector` or :func:`dolfinx.fem.petsc.assemble_vector`, depending on the value of ``petsc``.
#     Returns:
#         The assembled system vector ``l``.
        
#     .. note::
#         It might be advisable to call :meth:`merge_dirichletbcs` before this function.
#     """
    
#     test_space = l.function_spaces[0]
    
#     [trial_bc,test_bc] = merge_dirichletbcs(bcs, [trial_space, test_space])
    
#     if petsc:
#         l = _assemble_vector_PETSc(l, trial_bc, test_bc, **kwargs)
#     else:
#         l = _assemble_vector_scipy(l, trial_bc, test_bc, **kwargs)
#     return l


    
# def _assemble_matrix_scipy(A: fem.Form, trial_bc: DirichletCPPType, test_bc: DirichletCPPType, **kwargs) -> sparse.csr_array:
    
#     # naming convention trial=N, test=M, N/M: actual list of dofs, n/m: number of dofs
#     # total number of dofs / matrix dimensions
#     n = trial_bc.function_space.dofmap.index_map.size_global
#     m = test_bc.function_space.dofmap.index_map.size_global
#     # dofs restricted by dirichlet BCs
#     N_fixed,n_fixed = trial_bc.dof_indices()
#     M_fixed,m_fixed = test_bc.dof_indices()
#     # free dofs not resticted by dirichlet BCs
#     N_free = np.arange(n,dtype=np.int32); N_free = np.delete(N_free, N_fixed)
#     M_free = np.arange(m,dtype=np.int32); M_free = np.delete(M_free, M_fixed)
#     n_free = n - n_fixed
#     m_free = m - m_fixed
    
#     A = fem.assemble_matrix(A, **kwargs).to_scipy()
    
#     # no dirichlet BCs to apply
#     if n_fixed == 0 and m_fixed == 0:
#         return A
    
#     # rows to enforce the trial dirichlet values
#     A_diri = sparse.coo_array((np.ones(n_fixed), (np.arange(n_fixed), N_fixed)), shape=(n_fixed, n)).tocsr()
#     # add trial dirichlet rows and remove test dirichlet rows
#     return sparse.vstack([A_diri, A[M_free,:]])


# def _assemble_vector_scipy(l: fem.Form, trial_bc: DirichletCPPType, test_bc: DirichletCPPType, **kwargs) -> np.ndarray:
    
#     # naming convention trial=N, test=M, N/M: actual list of dofs, n/m: number of dofs
#     # total number of dofs / matrix dimensions
#     n = trial_bc.function_space.dofmap.index_map.size_global
#     m = test_bc.function_space.dofmap.index_map.size_global
#     # dofs restricted by dirichlet BCs
#     N_fixed,n_fixed = trial_bc.dof_indices()
#     M_fixed,m_fixed = test_bc.dof_indices()
#     # free dofs not resticted by dirichlet BCs
#     N_free = np.arange(n,dtype=np.int32); N_free = np.delete(N_free, N_fixed)
#     M_free = np.arange(m,dtype=np.int32); M_free = np.delete(M_free, M_fixed)
#     n_free = n - n_fixed
#     m_free = m - m_fixed
    
#     l = fem.assemble_vector(l, **kwargs).array
    
#     # no dirichlet BCs to apply
#     if n_fixed == 0 and m_fixed == 0:
#         return l
#     # set trial dirichlet values and remove test dirichlet rows
#     return np.concatenate((trial_bc.value.x.array[N_fixed], l[M_free]))


# def _assemble_matrix_PETSc(A: fem.Form, trial_bc: DirichletCPPType, test_bc: DirichletCPPType, **kwargs) -> PETSc.Mat:
    
#     # naming convention trial=N, test=M, N/M: actual list of dofs, n/m: number of dofs
#     # total number of dofs / matrix dimensions
#     n = trial_bc.function_space.dofmap.index_map.size_global
#     m = test_bc.function_space.dofmap.index_map.size_global
#     # dofs restricted by dirichlet BCs
#     N_fixed,n_fixed = trial_bc.dof_indices()
#     M_fixed,m_fixed = test_bc.dof_indices()
#     # free dofs not resticted by dirichlet BCs
#     N_free = np.arange(n,dtype=np.int32); N_free = np.delete(N_free, N_fixed)
#     M_free = np.arange(m,dtype=np.int32); M_free = np.delete(M_free, M_fixed)
#     n_free = n - n_fixed
#     m_free = m - m_fixed
    
#     A = dolfinx.fem.petsc.assemble_matrix(A, **kwargs)
#     A.assemble()
#     A_type = A.getType()
#     A_comm = A.comm
    
#     # no dirichlet BCs to apply
#     if n_fixed == 0 and m_fixed == 0:
#         return A
    
#     # ToDo: if len(trial_bc.fixed_dofs) == 0: just remove test dirichlet rows
#     # ToDO: if len(test_bc.fixed_dofs) == 0: just set trial dirichlet rows
#     # ToDo: if len(trial_bc.fixed_dofs) == len(test_bc.fixed_dofs): dont create a new matrix, modify A in place
    
#     # remove test dirichlet rows
#     is_rows = PETSc.IS().createGeneral(M_free, comm=A_comm)
#     is_col  = PETSc.IS().createGeneral(np.arange(n, dtype=np.int32), comm=A_comm)
#     A_free = A.createSubMatrix(is_rows, is_col)
#     A.destroy()
#     is_rows.destroy()
#     is_col.destroy()
    
#     # rows to enforce the trial dirichlet values
#     A_diri = PETSc.Mat().createAIJ(size=(n_fixed, n), nnz=1, comm=A_comm)
#     A_diri.setUp()
#     for i, j in enumerate(N_fixed): A_diri[i, j] = 1.0
    
#     # merge both matrices
#     A = PETSc.Mat().createNest([[A_diri],[A_free]], comm=A_comm)
#     A_diri.destroy()
#     A_free.destroy()
    
#     # restore original matrix type
#     A.assemble()
#     A.convert(A_type).assemble()
#     return A
    

# def _assemble_vector_PETSc(l: fem.Form, trial_bc: DirichletCPPType, test_bc: DirichletCPPType, **kwargs) -> PETSc.Vec:
    
#     # naming convention trial=N, test=M, N/M: actual list of dofs, n/m: number of dofs
#     # total number of dofs / matrix dimensions
#     n = trial_bc.function_space.dofmap.index_map.size_global
#     m = test_bc.function_space.dofmap.index_map.size_global
#     # dofs restricted by dirichlet BCs
#     N_fixed,n_fixed = trial_bc.dof_indices()
#     M_fixed,m_fixed = test_bc.dof_indices()
#     # free dofs not resticted by dirichlet BCs
#     N_free = np.arange(n,dtype=np.int32); N_free = np.delete(N_free, N_fixed)
#     M_free = np.arange(m,dtype=np.int32); M_free = np.delete(M_free, M_fixed)
#     n_free = n - n_fixed
#     m_free = m - m_fixed
    
#     l = dolfinx.fem.petsc.assemble_vector(l, **kwargs)
#     l.assemble()
    
#     # no dirichlet BCs to apply
#     if n_fixed == 0 and m_fixed == 0:
#         return l
    
#     # ToDo: if len(trial_bc.fixed_dofs) == 0: just remove test dirichlet rows
#     # ToDO: if len(test_bc.fixed_dofs) == 0: just set trial dirichlet rows
#     # ToDo: if len(trial_bc.fixed_dofs) == len(test_bc.fixed_dofs): dont create a new matrix, modify A in place
    
#     # create right hand side
#     l_ = PETSc.Vec().create(comm=l.comm)
#     l_.setType(l.getType())
#     l_.setSizes(n_fixed+m_free)
#     l_.setUp()
    
#     # set trial dirichlet values
#     l_.setValues(range(n_fixed), trial_bc.value.x.array[N_fixed])
    
#     # insert values that are not test dirichlet rows
#     is_rows = PETSc.IS().createGeneral(M_free, comm=l.comm)
#     l_diri = l.getSubVector(is_rows)
#     l_.setValues(range(n_fixed, n_fixed + m_free), l_diri.getArray())
#     l.restoreSubVector(is_rows, l_diri)
#     l.destroy()
#     is_rows.destroy()
    
#     l_.assemble()
#     return l_