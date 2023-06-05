import qutip, sys, pickle
import numpy as np
import scipy.optimize as opt 
import scipy.linalg as linalg

import a_quantum_stateology_lib as TpM
import b_quantum_geometries_lib as gij
import c_matrix_analysis_lib as matAnsys

# In [1]:

def solve_exact(Hamiltonian, K0, timespan, return_evolved_rhos = True,
                                           return_qutip_res_obj = True):
    
    res = qutip.mesolve(H = Hamiltonian, rho0 = K0, tlist = timespan,
                        c_ops = None, e_ops = None)
    if return_evolved_rhos: 
        rho_at_timet = [matAnsys.safe_expm_and_normalize(K = Kt1, return_free_energy = False, 
                                                                  tol = 1e-5) 
                           for Kt1 in res.states]
    return res, rho_at_timet
