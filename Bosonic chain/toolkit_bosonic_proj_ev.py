# In [1]:

import qutip
import numpy as np
import scipy.optimize as opt 
import pickle
import sys
import scipy.linalg as linalg

# In [2]:

def ev_checks(rho):
    a = bool; ev_list = linalg.eig(rho)[0]
    for i in range(len(ev_list)):
        if (ev_list[i] > 0):
            a = True
        else:
            a = False
            print("Eigenvalues not positive")
    return a

def is_density_op(rho):
    return (qutip.isherm(rho) and (rho.tr() == 1 or (1 - rho.tr() < 10**-10)) and ev_checks(rho)) 

# In [3]: 

def basic_bos_ops(L, dim):
    loc_globalid_list = []; loc_destroy_list = []; loc_create_list = []; loc_num_list = []
    
    id_op = qutip.qeye(dim)
    a_op = qutip.destroy(dim)
    adag_op = qutip.create(dim)
    num_op = qutip.num(dim)
    loc_global_id = qutip.tensor([qutip.qeye(dim) for k in range(L)])
    
    for n in range(L):
        operator_list = []
        for m in range(L):
            operator_list.append(id_op)
        loc_globalid_list.append(loc_global_id)
        operator_list[n] = a_op
        loc_destroy_list.append(qutip.tensor(operator_list))
        
        operator_list[n] = adag_op
        loc_create_list.append(qutip.tensor(operator_list))
        
        operator_list[n] = num_op
        loc_num_list.append(qutip.tensor(operator_list))
    return loc_globalid_list,loc_destroy_list,loc_create_list,loc_num_list

# In [4]: 

def all_one_and_two_body_bosonic_ops(L, dim):
    loc_globalid_list,loc_destroy_list,loc_create_list,loc_num_list = one_body_bosonic_ops(L, dim)
    
    four_vec = [loc_globalid_list,loc_destroy_list,loc_create_list]
    
    destroy_bos_op = []; create_bos_op = []; num_bos_op = []; two_body_bos = [];
    
    destroy_bos_op = [loc_destroy_list[n] * four_vec[a][b] for n in range(L)
                                                           for a in range(len(four_vec))
                                                           for b in range(len(four_vec[a]))]
    
    create_bos_op = [loc_create_list[n] * four_vec[a][b] for n in range(L)
                                                           for a in range(len(four_vec))
                                                           for b in range(len(four_vec[a]))]
    
    two_body_bos = [loc_globalid_list, destroy_bos_op, create_bos_op, loc_num_list]
    return two_body_bos

# In [5]: 

def bosonic_Hamiltonian(bosonic_system, L, dim, coeff, visualization):
    
    bos_system_type = ["harmonic oscillator", "tight-binding"]
    loc_globalid_list,loc_destroy_list,loc_create_list,loc_num_list = basic_bos_ops(L, dim)
    
    H = 0
    
    if (bosonic_system in bos_system_type):
        
        if bosonic_system == "harmonic oscillator":
            coeff = coeff * np.pi * np.ones(L)
            H += sum(coeff[l] * loc_num_list[l] for l in range(L))
            
        elif bosonic_system == "tight-binding":
            t = coeff
            H += -t * sum(loc_create_list[j] * loc_destroy_list[j+1] 
                          + loc_destroy_list[j] * loc_create_list[j+1] for j in range(L-2))
            H += -t * (loc_create_list[L-1] * loc_destroy_list[1] 
                       + loc_destroy_list[L-1] * loc_create_list[1])
    
    else:
        sys.exit("Not supported bosonic system")
    
    
    if visualization: 
        qutip.hinton(H)
        
    return H

# In [6]: 

def two_body_gaussian_state(L, dim, coeffs, visualization = False):
    loc_globalid_list,loc_destroy_list,loc_create_list,loc_num_list = basic_bos_ops(L, dim)
    
    a = len(basic_bos_ops(L, dim))
    b = len(basic_bos_ops(L, dim)[0])
    coeffs_list = coeffs * np.full((a,b),1)
    
    K = 0;
    
    K += sum(coeffs_list[a][b] * basic_bos_ops(L, dim)[a][b] 
             for a in range(L) for b in range(L))
    
    rho_loc = 0
    rho_loc = K.expm()
    rho_loc = rho_loc/ rho_loc.tr()
    
    if is_density_op(rho_loc):
        pass 
    else:  
        rho_loc = None 
        raise Exception("The result is not a density operator")
        
    if visualization: 
        qutip.hinton(rho_loc)
    
    return rho_loc

# In [7]: 

def prod_basis(b1, b2):
    return [qutip.tensor(b,s) for b in b1 for s in b2]

def scalar_prod(op1, op2, rho0 = None, HS_prod_modified = True):
    if op1.dims[0][0]==op2.dims[0][0]:
        pass
    else:
        raise Exception("Incompatible Qobj dimensions")
    if rho0 is None:
        rho0 = qutip.qeye(op1.dims[0])/op1.dims[0][0]
    if (HS_prod_modified):  
        result = .5*(rho0*(op1*op2.dag()+op2.dag()*op1)).tr()
    else: 
        result = .5*(rho0*(op1.dag()*op2)).tr()
    return result.real

def base_orth(ops, rho0):
    if isinstance(ops[0], list):
        ops = [op for op1l in ops for op in op1l]
    dim = ops[0].dims[0][0]
    basis = []
    for i, op in enumerate(ops): 
        alpha = [scalar_prod(op2, op, rho0) for op2 in basis]
        op_mod = op - sum([c*op2 for c, op2, in zip(alpha, basis)])
        op_norm = np.sqrt(scalar_prod(op_mod,op_mod,rho0))
        if op_norm<1.e-12:
            continue
        op_mod = op_mod/(op_norm)
        basis.append(op_mod)
    return basis

def logM(rho):
    if ev_checks(rho):
        pass
    else:
        raise Exception("Singular input matrix")
    eigvals, eigvecs = rho.eigenstates()
    return sum([np.log(vl)*vc*vc.dag() for vl, vc in zip(eigvals, eigvecs)])

def sqrtM(rho):
    if qutip.isherm(rho) and ev_checks(rho):
        pass
    else:
        raise Exception("Non-hermitian or singular input matrix")
    eigvals, eigvecs = rho.eigenstates()
    return sum([(vl**.5)*vc*vc.dag() for vl, vc in zip(eigvals, eigvecs)])

def proj_op(K, basis, rho0):
    return sum([scalar_prod(b, K,rho0) * b for b in basis])

def rel_entropy(rho, sigma):
    if (ev_checks(rho) and ev_checks(sigma)):
        pass
    else:
        raise Exception("Either rho or sigma non positive")
    
    val = (rho*(logM(rho)-logM(sigma))).tr()
                    
    if (abs(val.imag)>1.e-6):
        val = None
        raise Exception("Either rho or sigma not positive")
    return val.real
                
# In [11]:

def bures(rho, sigma):
    if is_density_op(rho) and is_density_op(sigma):
        val = abs((sqrtM(rho)*sqrtM(sigma)).tr())
        val = max(min(val,1.),-1.)
    return np.arccos(val)/np.pi
        
def maxent_rho(rho, basis):   
    def test(x, rho, basis):
        k = sum([-u*b for u,b in zip(x, basis)])        
        sigma = (.5*(k+k.dag())).expm()
        sigma = sigma/sigma.tr()
        return rel_entropy(rho, sigma)    
    res = opt.minimize(test,zeros(len(basis)),args=(rho,basis))
    k = sum([-u*b for u,b in zip(res.x, basis)])        
    sigma = (.5*(k+k.dag())).expm()
    sigma = sigma/sigma.tr()
    return sigma
 
def error_maxent_state(rho, basis, distance=bures):
    try:
        sigma = maxent_rho(rho, basis)
        return distance(rho,sigma)
    except:
        print("fail error max-ent state")
        return None
       
def error_proj_state(rho, rho0, basis, distance=bures):
    try:
        basis = base_orth(basis, rho0)
    except:
        print("orth error")
        raise
    try:
        sigma = proj_op(logM(rho), basis, rho0).expm()
        sigma = (sigma+sigma.dag())/(2.*sigma.tr())
    except:
        print("gram error")
    try:
        return distance(rho, sigma)
    except:
        print("fail error proj state")
        return None
    
# In [12]:

def spin_dephasing(N, gamma):
    loc_c_op_list = []; sz_list = one_body_spin_ops(N)[3];
    collapse_weights = abs(gamma) * np.ones(N)
    loc_c_op_list = [np.sqrt(collapse_weights[n]) * sz_list[n] for n in range(N)]
    return loc_c_op_list

# In [13]:
