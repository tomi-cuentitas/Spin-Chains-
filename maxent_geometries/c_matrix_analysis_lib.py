# In [1]:

import sys, pickle
import numpy as np
import scipy.optimize as opt 
import scipy.linalg as linalg

import qutip as Qobj
import a_quantum_stateology_lib as TpM 

# In [2]:

def safe_expm_and_normalize(K: Qobj, return_free_energy = True, tol = 1e-5):
    """
    Given a quantum observable K, lying in the tangent space of a Max-Ent manifold, this module maps it to a Max-Ent state sigma, s.t. the 
    mapping is smooth and s.t. sigma is a well-defined quantum state. 
    This module takes as input:
        
         *♥*♥* 1. K: a hermitian qutip.Qobj
         *♥*♥* 2. return_free_energy: boolean, default_value = True,
                                       This option, if toggled on, returns the free energy associated with the Max-Ent state.
         *♥*♥* 3. tol:                               
                                       
         
         ====> Returns: sigma = e^(K)/Tr(e^K).
         
         Warnings: (a)*. K must be a hermitian qutip.Qobj
                   (b)*. This module exponentiates K to a density state by avoiding overflows.
    """
    
    if TpM.non_hermitianess_measure(K) <= tol:
        pass
    else:
        print("Error: Non-hermitian input")
    
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
        f = None
    assert TpM.is_density_op(sigma), "sigma is not a density op"
    return sigma, f


    
    
    
    
    
    
    
