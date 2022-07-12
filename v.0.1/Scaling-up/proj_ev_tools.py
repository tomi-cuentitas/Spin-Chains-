#!/usr/bin/env python
# coding: utf-8

# In[1]:

import qutip
import numpy as np
import scipy.optimize as opt 
import pickle
import scipy.linalg as linalg

# In[2]:

id2 = qutip.qeye(2)
sx = .5*qutip.sigmax()
sy = .5*qutip.sigmay()
sz = .5*qutip.sigmaz()

def one_body_spin_ops(N = int):
    loc_sx_list = []
    loc_sy_list = []
    loc_sz_list = []
    loc_global_id = qutip.tensor([qutip.qeye(2) for k in range(N)])
    loc_globalid_list = []
    
    for n in range(N):
        operator_list = []
        for m in range(N):
            operator_list.append(id2)
        loc_globalid_list.append(loc_global_id)
        operator_list[n] = sx
        loc_sx_list.append(qutip.tensor(operator_list))
        
        operator_list[n] = sy
        loc_sy_list.append(qutip.tensor(operator_list))
        
        operator_list[n] = sz
        loc_sz_list.append(qutip.tensor(operator_list))        
    return loc_globalid_list, loc_sx_list, loc_sy_list, loc_sz_list

# In [3]:# updated version, lo pensé como un árbol orientado

# técnicamente hay 36 operadores en esta lista de listas para N = 2, pero hay muchos repetidos. 
# Ejemplo: sx^1 * sx^2 = - sx^2 * sx^1 y también sx^2 * sy^1 = sy^1 * sx^2

def all_two_body_spin_ops(N):
    loc_global_id_list = one_body_spin_ops(N)[0]
    sx_list = one_body_spin_ops(N)[1]
    sy_list = one_body_spin_ops(N)[2]
    sz_list = one_body_spin_ops(N)[3]
      
    pauli_four_vec = [loc_global_id_list, sx_list, sy_list, sz_list];
        
    sxsa_list = []; sysa_list = []; szsa_list = []; two_body_s = [];
    sxsa_list = [sx_list[n] * pauli_four_vec[a][b] for n in range(N)
                                              for a in range(len(pauli_four_vec))
                                              for b in range(len(pauli_four_vec[a]))]
    
    sysa_list = [sy_list[n] * pauli_four_vec[a][b] for n in range(N)
                                              for a in range(len(pauli_four_vec))
                                              for b in range(len(pauli_four_vec[a]))]
    
    szsa_list = [sz_list[n] * pauli_four_vec[a][b] for n in range(N)
                                             for a in range(len(pauli_four_vec))
                                             for b in range(len(pauli_four_vec[a]))]
    
    two_body_s = [sxsa_list, sysa_list, szsa_list]
    return two_body_s

def two_body_spin_ops(N):
    loc_ops = [];
    loc_global_id_list, sx_list, sy_list, sz_list = one_body_spin_ops(N)
    
    if (N == 2):
        loc_ops = all_two_body_spin_ops(N)
    else:
        loc_sxsx = [sx_list[n] * sx_list[m] for n in range(N)
                                            for m in range(N)]
        loc_sysy = [sy_list[n] * sy_list[m] for n in range(N)
                                            for m in range(N)]
        loc_szsz = [sz_list[n] * sz_list[m] for n in range(N)
                                            for m in range(N)]
        loc_ops.append(loc_sxsx)
        loc_ops.append(loc_sysy)
        loc_ops.append(loc_szsz)
    return loc_ops

# In [4]:

def Heisenberg_hamiltonian (N, Jx = list, Jy = list, Jz = list, h = list):
    globalid_list, sx_list, sy_list, sz_list = one_body_spin_ops(N)
    H = 0;
    for n in range(N):
        H += -0.5*h[n]*sz_list[n]
        
    for n in range(N-1):
        H += -0.5 * Jx[n] * sx_list[n] * sx_list[n+1]
        H += -0.5 * Jy[n] * sy_list[n] * sy_list[n+1]
        H += -0.5 * Jz[n] * sz_list[n] * sz_list[n+1]
    return H

# In [5]:

def free_particle_ops(N, H_H = 1, sz_list=list):
    loc_x_op = sum((.5 + sz_list[a])*(a+1) for a in range(N))
    loc_p_op = 1j * (loc_x_op*H_H - H_H*loc_x_op)
    loc_comm_xp = .5*(loc_x_op*loc_p_op + loc_p_op*loc_x_op)
    loc_corr_xp = -1j*(loc_x_op*loc_p_op - loc_p_op*loc_x_op)
    loc_p_dot = 1j*(H_H * loc_p_op - loc_p_op * H_H)
    return loc_x_op, loc_p_op, loc_comm_xp, loc_corr_xp, loc_p_dot

# In [6]:

natural = tuple('123456789')

def n_body_basis(gr, N):
    basis = []
    globalid_list, sx_list, sy_list, sz_list = one_body_spin_ops(N)
    
    if (isinstance(gr,int) and str(gr) in natural):
        if (gr == 1):
            basis = globalid_list + sx_list + sy_list + sz_list
        elif (gr > 1):
            basis = [op1*op2 for op1 in n_body_basis(gr-1, N) for op2 in n_body_basis(1, N)]
    else:
        basis = 'beep boop, gr must be natural'
    return basis

# In [7]: 

def basis_defs_test(N = 2):
    globalid_list, sx_list, sy_list, sz_list = one_body_spin_ops(N)
    
    one_site_basis = globalid_list + sx_list + sy_list + sz_list 
    
    two_site_basis = [op1*op2 for op1 in one_site_basis 
                              for op2 in one_site_basis]

    tercera_base = [op*op3 for op in two_site_basis for op3 in one_site_basis]
    three_site_basis = [op1*op2*op3 for op1 in one_site_basis 
                                    for op2 in one_site_basis 
                                    for op3 in one_site_basis]

    cuarta_base = [op*op4 for op in three_site_basis for op4 in one_site_basis]
    four_site_basis = [op1*op2*op3*op4 for op1 in one_site_basis 
                                       for op2 in one_site_basis 
                                       for op3 in one_site_basis 
                                       for op4 in one_site_basis]

    uno_a_cuatro_bases = []
    uno_a_cuatro_bases.append(one_site_basis)
    uno_a_cuatro_bases.append(two_site_basis)
    uno_a_cuatro_bases.append(three_site_basis)
    uno_a_cuatro_bases.append(four_site_basis)
    
    boolean = False 
    for n in range(4): 
        for m in range(3):
            if (n_body_basis(n+1, N) == uno_a_cuatro_bases[m] and n != m):
                print('Error: Two different index basis definitions equal', boolean)
            if (n_body_basis(n+1, N) != uno_a_cuatro_bases[m] and n == m):
                print('Error: Two same index basis definitions not equal', boolean)
            else:
                print('OK', True)
    return None

# In [6]:

def ev_checks(rho):
    a = bool 
    ev_list = linalg.eig(rho)[0]
    for i in range(len(ev_list)):
        if (ev_list[i] > 0):
            a = True
        else:
            a = False
            print("Eigenvalues not positive")
    return a

def is_density_op(rho):
    return (qutip.isherm(rho) and (rho.tr() == 1 or (1 - rho.tr() < 10**-10)) and ev_checks(rho)) 

# In [7]:

def n_body_max_ent_state(gr, N, coeffs = list):
    K = 0; rho_loc = 0;
    loc_global_id_list = one_body_spin_ops(N)[0]
    sx_list = one_body_spin_ops(N)[1];
    sy_list = one_body_spin_ops(N)[2];
    sz_list = one_body_spin_ops(N)[3];
    pauli_vec = [sx_list, sy_list, sz_list];
    
    if (gr == 1):
        try:
            K += sum(coeffs[n][m] *  one_body_spin_ops(N)[n][m] 
                                    for n in range(len(one_body_spin_ops(N)))
                                    for m in range(len(one_body_spin_ops(N)[n]))
                   ) 
        except Exception as ex:
            print(ex)
            raise ex
    elif (gr == 2): 
        try:
            K += sum(coeffs[n][m] * all_two_body_spin_ops(N)[n][m] 
                    for n in range(len(all_two_body_spin_ops(N)))
                    for m in range(len(all_two_body_spin_ops(N)[n]))
                   )
            K += loc_global_id_list[0]
        except Exception as ex:
            print(ex)
            raise ex
    else:
        print('gr must be either 1 or 2')
    
    rho_loc = K.expm()
    rho_loc = rho_loc/rho_loc.tr()
    
    if is_density_op(rho_loc):
        pass
    else:  
        rho_loc = None 
        print("The result is not a density operator")
        
    return rho_loc 

# In [7]: 

entropy_VN = []
def callback_entropy_VN (t,rhot):
    entropy_VN.append(qutip.entropy_vn(rhot))

# In[8]:

def prod_basis(b1, b2):
    return [qutip.tensor(b,s) for b in b1 for s in b2]

def scalar_prod(op1, op2, rho0 = None, HS_modified = True):
    if op1.dims[0][0]!=op2.dims[0][0]:
        return "Incompatible Qobj dimensions"
    if rho0 is None:
        rho0 = qutip.qeye(op1.dims[0])/op1.dims[0][0]
    if (HS_modified):  
        result = .5*(rho0*(op1*op2.dag()+op2.dag()*op1)).tr()
    else: 
        result = .5*(rho0*(op1.dag()*op2)).tr()
    result = result.real
    return result

def base_orth(ops, rho0):
    if isinstance(ops[0], list):
        ops = [op for op1l in ops for op in op1l]
      #print(type(ops),type(ops[0]))
    dim = ops[0].dims[0][0]
      #print("dim=",dim)
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
    eigvals, eigvecs = rho.eigenstates()
    return sum([np.log(vl)*vc*vc.dag() for vl, vc in zip(eigvals, eigvecs) if vl > 0])

def sqrtM(rho):
    eigvals, eigvecs = rho.eigenstates()
    return sum([(abs(vl)**.5)*vc*vc.dag() for vl, vc in zip(eigvals, eigvecs)])

def proj_op(K, basis, rho0):
    return sum([scalar_prod(b, K,rho0) * b for b in basis])

def rel_entropy(rho, sigma):
    val = (rho*(logM(rho)-logM(sigma))).tr()
    if abs(val.imag)>1.e-6:
        print("rho or sigma not positive")
        #print(rho.eigenstates())
        #print(sigma.eigenstates())
    return val.real

# In[9]:

#def qm_fidelity(rho, sigma):
#    vals = ((sqrtM(sqrtM(rho)*sqrtM(sigma)*sqrtM(rho))).tr())**2
#    vals = max(min(val, 1.), -1.
#    return vals

def bures(rho, sigma):
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

# In [10]:

def spin_dephasing(N, gamma):
    loc_c_op_list = []; sz_list = one_body_spin_ops(N)[3];
    collapse_weights = gamma * np.ones(N)
    for n in range(N):
        if collapse_weights[n] > 0.0:
            loc_c_op_list.append(np.sqrt(collapse_weights[n]) * sz_list[n])
    return loc_c_op_list

# In [11]:

def initial_state(N = 1, gaussian = True, gr = 1, x = .5, coeffs = list, psi0 = qutip.Qobj):
    loc_globalid = qutip.tensor([qutip.qeye(2) for k in range(N)]) 
    if gaussian: 
        rho0 = n_body_max_ent_state(gr, N, coeffs)
    else:
        if (qutip.isket(psi0)):
            rho0 = psi0 * psi0.dag()
            rho0 = x * rho0 + (1-x)*loc_globalid * x/N
            rho0 = rho0/rho0.tr()
        else:
            print("Psi0 must be a ket")
    return rho0     

# In [12]: legacy
## class one_body_ME():
##    def __init__(self, K):
##        self.K = linalg.expm(K)
##    def __repr__(self):
##        return self

## rho1 = one_body_ME(A)
## print(rho1)
