# In [0]:

import qutip
import numpy as np

# In [1]:

def ev_checks(rho, check_positive_definite = False, tol = 1e-5):
    """
    This module checks if a qutip.Qobj, a matrix, is either positive-definite or positive semi-definite, 
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
        if min_ev < tol:
            return False
        else:
            return True
    return True
        
def is_density_op(rho, verbose=False, critical=False, tol = 1e-3):
    """
    This module checks if the user-input QuTip.Qobj, rho, is a density operator or not. This is done 
    by checking if it is a hermitian, positive semi-definite, and trace-one, matrix. 
    This module takes as input the following parameters:
    
        *♥*♥* 1. rho: a qutip.Qobj,
        *♥*♥* 2. verbose: an optional boolean parameter for printing out logs,
                          stating which tests rho hasn't passed,
        *♥*♥* 3. critical: an optional boolean parameter.
        *♥*♥* 4. tol: an optional boolean parameter for establishing a maximum tolerance 
                        for numerical errors, when computing rho's trace. 
                        See Warnings further below.
       
        ====> Returns: a boolean, its truth value
                                  being whether or not 
                                  rho is a density matrix.
                                  
        Warnings: Due to numerical instabilities, it may be possible for the trace 
                  to not be exactly one, even though it is supposed to be. 
                  Therefore, a cut-off is implemented to check this condition.
    """
    if not qutip.isherm(rho):
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

# In [2]: 

def safe_expm_and_normalize(K, return_free_energy = False, tol = 1e-5):
    n_eig = sum(K.dims[0])
    if n_eig <= 16:
        e0 = max(K.eigenenergies())
    else:
        e0 = max(K.eigenenergies(sparse="True", sort = "high", eigvals = n_eig))
    sigma = (K-e0).expm()
    z = np.real(sigma.tr())
    sigma = sigma/z
    if return_free_energy: 
        f = -np.log(z) - e0;
    else:
        pass
    return sigma

def logM(rho, svd = True):
    """
    Evaluates the logarithm of a positive matrix rho.
    """
    assert ev_checks(rho), "Non positive-defined input matrix"
    if isinstance(rho, qutip.Qobj):
        qutip_form = True
        dims = rho.dims
    else:
        qutip_form = False        

    if svd:            
        if qutip_form:
            rho = rho.full()
        U, Sigma, Vdag = linalg.svd(rho, full_matrices = False)
        matrix_log = U @ np.diag(np.log(Sigma)) @ U.conj().transpose() 
    else: 
        if qutip_form:
            eigvals, eigvecs = rho.eigenstates()
            matrix_log = sum([np.log(vl)*vc*vc.dag() for vl, vc in zip(eigvals, eigvecs)]) 
        else:
            rho = rho.full()
            eigvals, eigvecs = linalg.eigh(rho)
            return evecs @ np.array([np.log(ev)*np.array(f) for ev, f in zip(evals, evecs.transpose().conj())])
    
    if qutip_form:
        matrix_log = qutip.Qobj(matrix_log, dims)
    return matrix_log

def sqrtM(rho, svd = True):
    """
    Evaluates the square root of a positive matrix rho.
    """
    assert ev_checks(rho), "Non positive-defined input matrix"
    if isinstance(rho, qutip.Qobj):
        qutip_form = True
        dims = rho.dims
    else:
        qutip_form = False

    if svd:
        if qutip_form:
            rho = rho.full()
        U, Sigma, Vdag = linalg.svd(rho, full_matrices = False)
        matrix_log = U @ np.diag((Sigma)**.5) @ U.conj().transpose() 
    else: 
        if qutip_form:
            eigvals, eigvecs = rho.eigenstates()
            matrix_log = sum([(vl**.5)*vc*vc.dag() for vl, vc in zip(eigvals, eigvecs)]) 
        else:
            rho = rho.full()
            eigvals, eigvecs = linalg.eigh(rho)
            return evecs @ np.array([(ev**.5)*np.array(f) for ev, f in zip(evals, evecs.transpose().conj())])
    
    if qutip_form:
        matrix_log = qutip.Qobj(matrix_log, dims)
    return matrix_log

# In [3]: 

def bures(rho, sigma, svd = True):
    """
    Evaluates the Bures metric between two density states. 
    """
    assert is_density_op(rho), "rho is not a density operator"
    assert is_density_op(sigma), "sigma is not a density operator"
    
    sqrt_sigma = sqrtM(sigma.full(), svd=svd)
    
    fidelity = sqrtM((sqrt_sigma @ rho.full()  @sqrt_sigma),svd=True).trace().real
​
    assert abs(fidelity.imag)<1.e-10, f"complex fidelity? fidelity={fidelity}"
    fidelity = fidelity.real
    assert 0 <= fidelity, f"negative fidelity? fidelity={fidelity}"
    if fidelity>1.05:
        assert (fidelity-1)<1.e-8, f"error in fidelity too large fidelity={fidelity}"
        return 0.
    return  np.arccos(fidelity)/np.pi
​
def rel_entropy(rho, sigma, svd = True):
    if svd:
        val = (rho*(logM(rho, True) - logM(sigma, True))).tr()
    else:
        assert ((ev_checks(rho) and ev_checks(sigma))), "Either rho or sigma have negative ev."
        val = (rho*(logM(rho, False)-logM(sigma, False))).tr()
        if (abs(val.imag - 0)>1.e-10):
            val = None
            raise Exception("Either rho or sigma have negative ev.")
    return val.real
def mean_field_iteration(rho0, guesstimates, basis, steps):
    trial_K = sum(phi * op for phi,op in zip(guesstimates, basis))
    sigma0 = safe_expm_and_normalize(trial_K)
    mft_states = [sigma0]
    for i in range(1, steps):
        sigma_mft = mft_states[i-1]
        coeff_list_j = [qutip.expect(sigma_mft, op) for op in basis]
        K_j = sum(zeta_j_i * op_i for zeta_j_i, op_i in zip(coeff_list_j, basis))
        sigma_j = safe_expm_and_normalize(K_j)
        mft_states.append(sigma_j)
        coeff_list_j = K_j = sigma_j = None
        
    return mft_states
