# In [0]:

import qutip, sys, pickle
import numpy as np
import scipy.optimize as opt 
import scipy.linalg as linalg

# In [1]:
## HS geometry 

def HS_inner_prod(op1, op2): 
    """
    This module computes the real-valued Hilbert-Schmidt inner product between 
    two operators. This module takes as input:
        *♥*♥* 1. op1, op2: two qutip.Qobj operators,
                 
        ====> Returns: Tr (op1.dag() * op2)
              
              Warnings: (a). The inputs must all be QuTip.Qobj
                        (b). An exception will be raised if the input operators have non-compatible
                             dimensions.
    
    """
    if (op1.dims[0][0]==op2.dims[0][0]):   
        pass                                
    else:                                   
        raise Exception("Incompatible Qobj dimensions")
    return (op1.dag() * op2).tr()

# In [2]:
## Static and Dynamical Correlation geometries

def correlation_inner_prod(op1, op2, rhoref = None): ### This inner product is real valued, provided both op1 and op2 are hermitian
    """
    This module computes the real-valued correlation inner product between 
    two operators, with an additional parameter for establishing non-equal statistical weights to 
    the operators themselves, namely a referential matrix.
    This module takes as input:
        *♥*♥* 1. op1, op2: two qutip.Qobj operators,
        *♥*♥* 3. rhoref: a referential density operator, which prescribes non-equal statistical weights 
                 to different operators, in the algebra of observables of a quantum system.
                 
        ====> Returns: Tr (rhoref * (op1.dag * op2 + op2 * op1.dag() ))
              
              Warnings: (a). The inputs must all be QuTip.Qobj
                        (b). An exception will be raised if the input operators have non-compatible
                             dimensions.
    
    """
    assert (op1.dims[0][0]==op2.dims[0][0]), "Incompatible Qobj dimensions"
    if rhoref is None:
        rhoref = qutip.qeye(op1.dims[0])
        rhoref = rhoref/rhoref.tr()
    else:
        assert is_density_op(rhoref), "rhoref is not a density op" 
    return .5 * (rhoref * anticommutator(op1.dag(), op2)).tr()

# In [3]:
## Kubo geometry

def integrand_Kubo_inner_prod(sigma):
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

def fetch_kubo_int_scalar_product(sigma:Qobj):
    """
    Build a KMB scalar product function
    associated to the state `sigma`, from
    its intergral form.
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













