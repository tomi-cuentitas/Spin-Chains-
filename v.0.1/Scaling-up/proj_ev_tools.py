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

def one_body_spin_ops(N):
    loc_sx_list = []; loc_sy_list = []; loc_sz_list = []; loc_globalid_list = []
    id2 = qutip.qeye(2)
    sx = .5*qutip.sigmax()
    sy = .5*qutip.sigmay()
    sz = .5*qutip.sigmaz()
    loc_global_id = qutip.tensor([qutip.qeye(2) for k in range(N)])
    
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

# In [4]:

def all_two_body_spin_ops(N):
    loc_global_id_list, sx_list, sy_list, sz_list = one_body_spin_ops(N)
      
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

# In [5]: 

def two_body_spin_ops(N, build_all = False):
    loc_list = []
    if build_all:
        loc_list = all_two_body_spin_ops(N)
    else: 
        loc_global_id_list, sx_list, sy_list, sz_list = one_body_spin_ops(N)        
        loc_sxsx = []; loc_sysy = []; loc_szsz = [];
        
        loc_sxsx = [sx_list[n] * sx_list[m] for n in range(N)
                                            for m in range(N)]
        loc_sysy = [sy_list[n] * sy_list[m] for n in range(N)
                                            for m in range(N)]
        loc_szsz = [sz_list[n] * sz_list[m] for n in range(N)
                                            for m in range(N)]
        loc_list.append(loc_sxsx)
        loc_list.append(loc_sysy)
        loc_list.append(loc_szsz)
    return loc_list

# In [6]: 

def Heisenberg_Hamiltonian(chain_type, N, visualization, Jx, Jy, Jz, h):
    spin_chain_type = ["XYZ", "XXZ", "XXX"]
    loc_global_id_list, sx_list, sy_list, sz_list = one_body_spin_ops(N)   
      
    H = 0
    
    Jx = Jx * 2 * np.pi * np.ones(N)
    h = h * 2 * np.pi * np.ones(N)
    H += sum(-.5* h[n] * sz_list[n] for n in range(N))
    
    if (chain_type in spin_chain_type): 
        if chain_type == "XXX":
            H += sum(-.5* Jx[n] * (sx_list[n]*sx_list[n+1] 
                                 + sy_list[n]*sy_list[n+1]
                                 + sz_list[n]*sz_list[n+1]) for n in range(N-1))
        
        elif chain_type == "XXZ":
            Jz = Jz * 2 * np.pi * np.ones(N)
            H += sum(-.5 * Jx[n] * (sx_list[n] * sx_list[n+1] + sy_list[n] * sy_list[n+1]) 
                     -.5 * Jz[n] * (sz_list[n] * sz_list[n+1]) for n in range(N-1))
        
        elif chain_type == "XYZ":
            Jy = Jy * 2 * np.pi * np.ones(N)
            Jz = Jz * 2 * np.pi * np.ones(N)
            H += sum(-.5 * Jx[n] * (sx_list[n] * sx_list[n+1])
                     -.5 * Jy[n] * (sy_list[n] * sy_list[n+1]) 
                     -.5 * Jz[n] * (sz_list[n] * sz_list[n+1]) for n in range(N-1))
    else:
        sys.exit("Currently not supported chain type")
              
    if visualization:
        qutip.hinton(H)
              
    return H
    
# In [7]:

def classical_ops(chain_type, N, Jx, Jy, Jz, h):
    H_H = Heisenberg_Hamiltonian(chain_type, N, False, Jx, Jy, Jz, h)
    sz_list = one_body_spin_ops(N)[3]
    
    loc_x_op = sum((.5 + sz_list[a])*(a+1) for a in range(N))
    loc_p_op = 1j * (loc_x_op*H_H - H_H*loc_x_op)
    loc_comm_xp = .5*(loc_x_op*loc_p_op + loc_p_op*loc_x_op)
    loc_corr_xp = -1j*(loc_x_op*loc_p_op - loc_p_op*loc_x_op)
    loc_p_dot = 1j*(H_H * loc_p_op - loc_p_op * H_H)
    
    return loc_x_op, loc_p_op, loc_comm_xp, loc_corr_xp, loc_p_dot
   
# In [8]: 

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

# In [9]:

def n_body_max_ent_state(gr, N, coeffs = list, visualization = False):
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
        except Exception as exme1:
            print(exme1, "Max-Ent 1 Failure")
            raise ex
    elif (gr == 2): 
        try:
            K += sum(coeffs[n][m] * all_two_body_spin_ops(N)[n][m] 
                    for n in range(len(all_two_body_spin_ops(N)))
                    for m in range(len(all_two_body_spin_ops(N)[n]))
                   )
            K += loc_global_id_list[0]
        except Exception as exme2:
            print(exme2, "Max-Ent 2 Failure")
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
        
    if visualization: 
        qutip.hinton(rho_loc)
        
    return rho_loc 
