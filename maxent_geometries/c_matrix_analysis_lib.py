# In [1]:

import sys, pickle
import numpy as np
import scipy.optimize as opt 
import scipy.linalg as linalg

import qutip as Qobj
import a_quantum_stateology_lib as TpM 
import b_quantum_geometries_lib as gij

# In [1]:

def spectral_norm(op):
    linalg.norm(op, np.inf)

def gram_matrix(basis: list, innnerprod: callable, as_qutip_qobj = True):
    """
    This module computes the Gram matrix associated to a basis of observables, for a quantum system, and a specific inner product.
    This module takes as input:
        *♥*♥* 1. basis: a list of quantum observables, in qutip.Qobj format.
        *♥*♥* 2. innerprod: the induced inner product, on the space of observables.
        *♥*♥* 3. (optional) as_qutip_qobj: boolean, default value: True.
                                            A boolean option which, if toggled on, returns the Gram matrix as a qutip.Qobj
        
        ====> Returns: the Gram matrix as an np.array.
        
        Warnings: (*). all entries must be square matrices. 
    
    """
    size = len(basis) 
    gram = np.array([[innerprod(op1, op2).round(14) for op2 in basis] for op1 in basis])
    if as_qutip_qobj:
        gram = qutip.Qobj(gram)
    return gram
    
def orthogonalize_basis(basis: list, innerprod: callable, idop: Qobj = None, tol = 1e-5):
    """
    This module orthonormalizes
        *♥*♥* 1. sigma: a qutip.Qobj, namely a quantum state. 
        
        ====> Returns: a lambda function, given in terms of the state's eigenvalues and eigenvectors. 
    
    """
    local_basis = basis
    if idop:
        sqidnorm = innerprod(idop, idop)
        id_comp = [innerprod(idop, op)/sqidnorm for op in basis]
        local_basis = [idop * sqidnorm ** (-.5)] + [op - mu for mu, op in zip(id_comp, basis)]
    
    gram = gram_matrix(local_basis, innerprod, False)
    evals, evecs = np.linalg.eigh(gram)
    evecs = [vec/np.linalg.norm(vec) for vec in evecs.transpose()]
    return [
            mu ** (-.5) * sum( c* op for c,op in zip(v, basis))
                for mu, v in zip(evals, evecs) if p >= tol
        ]

# In [2]: 

def project_op(op, orth_basis, scalar_prod: callable):
    """
    This module computes
        *♥*♥* 1. sigma: a qutip.Qobj, namely a quantum state. 
        
        ====> Returns: a lambda function, given in terms of the state's eigenvalues and eigenvectors. 
    
    """
    return np.array([innerprod(qop, op) for qop in orth_basis])

def static_projection(K_at_timet, orth_basis, scalar_prod):
    distance = gij.fetch_induced_distance(scalar_prod)
    phi_vector = project_op(op = K_at_timet, orth_basis = orth_basis, scalar_prod = scalar_prod)
    Kp = sum(phi * qop for phi, qop in zip(phi_vector, orth_basis))
    return distance(Kp, K_at_timet), spectral_norm(Kp - K_at_timet), distance(K_at_timet,0)
                                            
def instantaneous_proj(K_at_timet, rho_at_timet, orth_basis, fetch_scalar_prod):
    scalar_prod = fetch_scalar_prod(rho_at_timet)
    distance = gij.fetch_induced_distance(scalar_prod)
    phi_vector = project_op(op = K_at_timet, orth_basis = orth_basis, scalar_prod = scalar_prod)
    Kp = sum(phi * qop for phi, qop in zip(phi_vector, orth_basis))
    return distance(Kp, K_at_timet), spectral_norm(Kp - K_at_timet), distance(K_at_timet,0)

# In [3]:

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

def build_Hierarchical_Basis(Hamiltonian, seed_operator, depth, tol = 1e-5, verbose = False):
    assert linalg.norm(seed_operator - seed_operator.dag()), "Error: Seed operator not Hermitian"
    HBasis = [seed_operator]
    for i in range(depth):
        local_op = 1j * gij.commutator(Hamiltonian, HBasis[i-1])
        assert linalg.norm(local_op - local_op.dag()) < tol, "Error: Iterated Commutator not Hermitian"
        norm = linalg.norm(local_op)
        if norm > tol:
            pass
        else: 
            local_op = None
            if verbose:
                print("     ###. HBasis terminated at step ", i)
        Hbasis.append(local_op)
        local_op = norm = None
    return Hbasis   
        
    
