# In [1]:

import qutip, sys, pickle
import numpy as np
import scipy.optimize as opt 
import scipy.linalg as linalg

# In [2]:

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
        ev_list = rho.eigenenerges()
        ev_list = sorted(rho.eigenenergies())
        min_ev = min(ev_list); ev_list = None
        if min_ev < 0:
            assert abs(min_ev) > tol, "Qobj is not positive semi-definite"
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

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    