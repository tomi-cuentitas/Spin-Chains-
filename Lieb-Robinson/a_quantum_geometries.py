# In [1]:

import numpy as np
import scipy.linalg as linalg
import qutip

from qutip import Qobj
from typing import Callable, List, Optional, Tuple, Dict
from scipy.integrate import odeint, solve_ivp

# In [2]:

check_hermitian= lambda rho,tol=1e-5: linalg.norm(rho - rho.dag())<=tol

def ev_checks(rho, check_positive_definite = False, tol = 1e-5):
    """
    This module checks whether a qutip.Qobj, a matrix, is either positive-definite or positive semi-definite, 
    i.e. whether or not all of its eigenvalues are strictly positive or non-negative, respectively. 
    This module takes as input the following parameters:
    
        *♥*♥* 1. rho: a qutip.Qobj.
        *♥*♥* 2. check_positive_definite: an optional boolean parameter 
                                            If it is toggled on, positive-definiteness 
                                            will be checked via a Cholesky decomposition.
                                          Otherwise, only positive semi-definiteness 
                                            will be analyzed, by explicitly checking the 
                                            matrix's eigenvalues. 
        *♥*♥* 3. tol: an optional boolean parameter, used only when positive semi-definiteness
                       is being analyzed, by establishing an upper bound on the smallest eigenvalue. 
        
        ====> Returns: a boolean, its truth value
                                  being whether or not 
                                  rho is a positive-definite
                                  matrix. 
        Warnings: None.
    """
    if isinstance(rho, qutip.Qobj):
        pass
    else:
        rho = qutip.Qobj(rho)
    
    if check_positive_definite: 
        try:
            rho = rho.full()
            np.linalg.cholesky(rho)
        except:
            return False
        return True
    else:
        ev_list = rho.eigenenergies()
        ev_list = sorted(rho.eigenenergies())
        min_ev = min(ev_list); ev_list = None
        if min_ev < 0:
            print( abs(min_ev) > tol, "Qobj is not positive semi-definite" )
    return True
    
def is_density_op(rho, verbose=False, critical=False, tol = 1e-5):
    """
    This module checks if the user-input QuTip.Qobj, rho, is a density operator or not. This is done by checking if it is a hermitian, 
    positive semi-definite, and trace-one, matrix. This module takes as input the following parameters:
    
        *♥*♥* 1. rho: a qutip.Qobj,
        *♥*♥* 2. verbose: an optional boolean parameter for printing out logs,
                          stating which tests rho hasn't passed,
        *♥*♥* 3. critical: an optional boolean parameter
                           ???,
        *♥*♥* 4. tol: an optional boolean parameter for establishing a maximum tolerance 
                      for numerical errors, when computing rho's trace. See Warnings further below.
       
        ====> Returns: a boolean, its truth value being whether or not rho is a density matrix.
                                  
        Warnings: Due to numerical instabilities, it may be possible for the trace 
                  to not be exactly one, even though it is supposed to be. 
                  Therefore, a cut-off is implemented to check for 
    """
    if not qutip.isherm(rho):
        if (np.linalg.norm(rho - rho.dag()) < tol):
            return True
        else:
            if verbose:
                print("rho is not hermitian")
            assert not critical
            return False
    if abs(1 - rho.tr()) > tol:
        if verbose:
            print("Tr rho != 1, Tr rho = ", rho.tr())
        assert not critical
        return False
    if not ev_checks(rho):
        if verbose:
            print("rho is not positive")
        assert not critical
        return False
    return True    

# In [3]:

def safe_expm_and_normalize(K) -> Qobj:
    """
    Compute the matrix exponential and normalize the result for a given operator K.

    Parameters:
    - K (Qobj): Quantum object representing the input operator.

    Returns:
    - Qobj: Quantum object representing the normalized matrix exponential of the input operator.

    This function calculates the matrix exponential of the input operator `K` and normalizes
    the result to ensure the trace of the resulting matrix is equal to 1. The normalization
    is particularly important in quantum mechanics to maintain the probabilistic interpretation
    of quantum states.

    The function uses an efficient method to compute the matrix exponential depending on the
    number of eigenvalues in the operator. If the number of eigenvalues is less than or equal
    to 16, the eigenenergies are directly used. Otherwise, a sparse eigenvalue calculation
    is performed using the `eigenenergies` function with the options for sparse matrix handling.

    Parameters:
    - K (Qobj): Quantum object representing the input operator.

    Returns:
    - Qobj: Quantum object representing the normalized matrix exponential of the input operator.

    Example:
    ```
    import qutip
    import numpy as np

    # Create a test operator K
    K = qutip.Qobj(np.random.rand(4, 4))

    # Calculate the safe matrix exponential and normalize
    result = safe_expm_and_normalize(K)
    ```

    Note: This function assumes that the input operator `K` is a valid quantum object (Qobj) from
    the QuTiP library.
    """
    n_eig = sum(K.dims[0])
    if n_eig <= 16:
        e0 = max(np.real(K.eigenenergies()))
    else:
        e0 = max(np.real(K.eigenenergies(sparse="True", sort="high", eigvals=n_eig)))
    sigma = (K - e0).expm()
    local_sigma_tr = np.real(sigma.tr())
    if local_sigma_tr > 1e-5:
        sigma=sigma/sigma.tr()
    return sigma
    
# In [4]:

def commutator(op1,op2) -> Qobj:
    return(op1*op2 - op2*op1)

def anticommutator(op1,op2) -> Qobj:
    return(op1*op2+op2*op1)
    
def fetch_covar_scalar_product(sigma): 
    return lambda op1,op2: .5*(sigma * anticommutator(op1.dag(), op2)).tr()

# In [5]: 
    
def basis_hermitian_check(basis):
    """
    Given a list or a dictionary of square Qutip.Qobj operators
    this module computes the non-hermitian character of said 
    operators by calculating the Frobenius norm of an operator
    and its adjoint. 
    
    This module takes as input the following parameters:
        *♥*♥* 1. basis: a length-M list of square Qutip.Qobj 
                        operators,
        ====> Returns: a length-M list of real-valued numbers,
                       where the i-th number is the following
                       Frobenius norm
                       
                       || basis[i] - basis[i].dag() ||_F
        
        Warnings: (a). This module assumes that the basis contains
                       square-matrices only. 
                  (b). If a dictionary is received, this module will 
                       extract its values, without changing the 
                       original basis' type.
    """
    if type(basis) is dict:
        basis_loc = basis.values()
    return [null_matrix_check(op - op.dag()) for op in basis_loc]

def gram_matrix(basis: List[Qobj], sp: Callable):
    size = len(basis)
    result = np.zeros([size, size], dtype=float)

    for i, op1 in enumerate(basis):
        for j, op2 in enumerate(basis):
            if j < i:
                continue
            entry = np.real(sp(op1, op2))
            if i == j:
                result[i, i] = entry
            else:
                result[i, j] = result[j, i] = entry

    return result.round(14)

# In [6]:

def orthogonalize_basis(basis: List[Qobj], sp: callable, idop = None, tol = 1e-5):
    local_basis = basis
    if idop:
        sqidnorm = sp(idop, idop)
        id_comp = [sp(idop, op)/sqidnorm for op in basis]
        local_basis = [idop * sqidnorm ** (-.5)] + [op - mu for mu, op in zip(id_comp, basis)]
    
    gram = gram_matrix(local_basis, sp)
    evals, evecs = np.linalg.eigh(gram)
    evecs = [vec/np.linalg.norm(vec) for vec in evecs.transpose()]
    local_basis=[mu ** (-.5) * sum( c* op for c,op in zip(v, local_basis)) for mu, v in zip(evals, evecs) if mu>tol]
    
    assert linalg.norm(gram_matrix(basis=local_basis, sp=sp) - np.identity(len(local_basis)))<tol, "Error: Basis not correctly orthogonalized"
    return local_basis

def proj_op(K: Qobj, basis: List[Qobj], sp: Callable):
    orth_basis=orthogonalize_basis(basis=basis, sp=sp)
    phi_mft=np.array([sp(Q, K) for Q in orth_basis])
    K_mft=sum(phis_mft_a * Q for phis_mft_a, Q in zip(phi_mft, orth_basis))
    return phi_mft, K_mft

def Kstate_from_phi_basis(phi: np.array, basis: List[Qobj]):
    if len(phi) < len(basis):
        phi=np.array(list(phi)+[.0 for i in range(len(basis) - len(phi))])
    return -sum(phi_a*opa for phi_a, opa in zip(phi, basis))

# In [7]: 

def build_HierarchicalBasis(generator: Qobj, seed_operator: Qobj, depth: int, tol = 1e-5, verbose = False):
    assert linalg.norm(seed_operator - seed_operator.dag()) < tol, "Error: Seed operator not Hermitian"
    hierarch_basis_local = [seed_operator]
    for i in range(1, depth+1):
        local_op = 1j * commutator(generator, hierarch_basis_local[i-1])
        assert linalg.norm(local_op - local_op.dag()) < tol, "Error: Iterated Commutator not Hermitian"
        norm = linalg.norm(local_op)
        if norm > tol:
            pass
        else: 
            local_op = None
            if verbose:
                print("     ###. HBasis terminated at step ", i)
        hierarch_basis_local.append(local_op)
        local_op = norm = None
    return hierarch_basis_local   
    
def Hij_tensor(basis: List[Qobj], sp: Callable, generator: Qobj):
    local_Hij = np.array([[sp(op1, commutator(-1j*generator, op2))
                            for op2 in basis] for op1 in basis])
    return np.real(local_Hij)

# In [8]:

def magnus_1t(generator, args):
    period=args.get('period')
    local_timespan_period=np.linspace(0,period, int(period)*50)
    local_magnus=sum(generator(t=t1, args=args) for t1 in local_timespan_period)
    return 1/period*local_magnus
    
def magnus_2t(generator, args):
    period=args.get('period')
    local_timespan_period=np.linspace(0,period, int(period)*100)
    local_magnus=0
    for t in local_timespan_period:
        for tprime in local_timespan_period:
            if tprime <= t:  
                local_magnus+=commutator(generator(t=t, args=args), generator(t=tprime, args=args))
    return 1/(2*1j*period)*local_magnus*(local_timespan_period[1]-local_timespan_period[0])**2
    
