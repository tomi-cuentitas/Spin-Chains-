# In [0]:

import sys, pickle
import numpy as np
import scipy.optimize as opt 
import scipy.linalg as linalg

import qutip as Qobj

# In [1]:

def commutator(A, B):
    """
    Given two square matrices, operators, this module computes its commutator. 
    This module takes as input the following parameters:
        *♥*♥* 1. A: a complex-valued matrix,
        *♥*♥* 2. B: another complex valued matrix.
        ====> Returns: A*B - B*A
        
        Warnings: This module first checks the compatibility of the matrix dimensions. 
                  Qutip.Qobj formats needed for both matrices.
    """
    if A.dims[0][0] == B.dims[0][0]: 
        pass
    else:
        raise Exception("Incompatible Qobj dimensions")
    return A*B - B*A

def anticommutator(A, B):
    """
    Given two square matrices, operators, this module computes its anticommutator. 
    This module takes as input the following parameters:
        *♥*♥* 1. A: a complex-valued matrix,
        *♥*♥* 2. B: another complex valued matrix.
        ====> Returns: A*B + B*A
        
        Warnings: This module first checks the compatibility of the matrix dimensions. 
                  Qutip.Qobj formats needed for both matrices.
    """
    if A.dims[0][0] == B.dims[0][0]: 
        pass
    else:
        raise Exception("Incompatible Qobj dimensions")
    return A*B+B*A

# In [2]:
## HS geometry

def fetch_HS_inner_prod(): 
    """
    This module instantiates a lambda function, associated to the Hilbert Schmidt inner product.
                 
        ====> Returns: Tr (op1.dag() * op2)
              
              Warnings: (a). The inputs must all be QuTip.Qobj
                        (b). An exception will be raised if the input operators have non-compatible
                             dimensions.
    """
    return lambda op1, op2: .5 * (op1.dag() * op2).tr()

# In [3]:
## static and dynamical correlation product  

def fetch_corr_scalar_prod(sigma:Qobj):
    """
    This module instantiates a lambda function, associated to the (static or dynamical) correlation scalar product.
    This correlation scalar product is calculated for a specific sigma-state, belonging to the family of correlation scalar products. 
    This module takes as input:
        *♥*♥* 1. sigma: a referential density operator, which prescribes possibly non-equal statistical weights 
                         to different operators, in the algebra of observables of a quantum system.
             
        ====> Returns: Tr (sigma * anticommutator(op1.dag() * op2))
        
        Warnings: (a). The inputs must all be QuTip.Qobj
    """
    return lambda op1, op2: .5 * (sigma * (anticommutator(op1.dag(), op2) ).tr())
                                  
# In [4]:
## Kubo geometry 

def Kubo_inner_prod_integrand(sigma):
    """
    This module instantiates the (integrand) of a specific sigma-weighted inner product, belonging to the family of state-dependent KMB 
    inner products, associated to the sigma state. This module takes as input:
        *♥*♥* 1. sigma: a qutip.Qobj, namely a quantum state. 
        
        ====> Returns: a lambda function, given in terms of the state's eigenvalues and eigenvectors. 
        
              Warnings: (a). sigma must be a qutip.Qobj
                        (b). sigma must be a valid quantum state. 
                        
    """
    sigma = qutip.Qobj(sigma)    
    evals, evecs = local_sigma.eigenstates()
    return lambda op1, op2: sum(
        (
            np.conj((v2.dag() * op1 * v1).tr())
            * ((v2.dag() * op2 * v1).tr())
            * (p1 if p1 == p2 else (p1 - p2) / np.log(p1 / p2))
        )
        for p1, v1 in zip(evals, evecs) 
        for p2, v2 in zip(evals, evecs)
    )

def fetch_Kubo_inner_prod(sigma):
    """
    This module computes a KMB inner product function, associated to the sigma state, from its integral form. 
    This module takes as input:
        *♥*♥* 1. sigma: a qutip.Qobj, namely a quantum state. 
        
        ====> Returns: a lambda function, given in terms of the state's eigenvalues and eigenvectors. 
    
    """
    evals, evecs = sigma.eigenstates()
    return lambda op1, op2: 0.01 * sum(
        (
            np.conj((v2.dag() * op1 * v1).tr())
            * ((v2.dag() * op2 * v1).tr())
            * p1 ** (1 - tau)
            * p1 ** (tau)
        )
        for p1, v1 in zip(evals, evecs)
        for p2, v2 in zip(evals, evecs)
        for tau in np.linspace(0, 1, 100)
    )

def fetch_induced_geo_distance(innerprod):
    local_dop = op1 - op2
    return np.sqrt(innerprod(local_dop, local_dop))
    
    
