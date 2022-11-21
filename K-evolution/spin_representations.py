# In [1]:

import qutip, sys, pickle
import numpy as np
import scipy.optimize as opt 
import scipy.linalg as linalg

import matrix_analysis_lib as mat_ansys

# In [2]:

def prod_basis(b1, b2):
    """
    This module constructs the tensor product of two
    operators. It takes as input:
        
    ***. two qutip.Qobjs
    
    """
    return [qutip.tensor(b,s) for b in b1 for s in b2]

def one_body_spin_ops(size):
    """
    Given an N-site spin chain, there are then 3N different, 
    non-trivial, operators acting on the full Hilbert space.
    N sigma-x operators, N sigma-y operators, 
    N sigma-z operators, and a global identity operator. 
    All these 3N+1-operators are constructed with a tensor
    product so that they all act on the full Hilbert space. 
    All but the global identity operator, act 
    non-trivially only on one Hilbert subspace. 
    """
    ### Basic, one-site spin operators are constructed.
    
    loc_sx_list = []; loc_sy_list = []; loc_sz_list = []; loc_globalid_list = []
    id2 = qutip.qeye(2)
    sx = .5*qutip.sigmax()
    sy = .5*qutip.sigmay()
    sz = .5*qutip.sigmaz()
    
    ### The global identity operator is constructed 
    loc_global_id = [qutip.tensor([qutip.qeye(2) for k in range(size)])]
    
    ### Lists of one-body operators are constructed, so that they all act on the full Hilbert space. 
    ### This is done via taking tensor products on lists of operators. 
    
    for n in range(size):
        operator_list = []
        for m in range(size):
            operator_list.append(id2)
        loc_globalid_list.append(loc_global_id)
        operator_list[n] = sx
        loc_sx_list.append(qutip.tensor(operator_list))
        
        operator_list[n] = sy
        loc_sy_list.append(qutip.tensor(operator_list))
        
        operator_list[n] = sz
        loc_sz_list.append(qutip.tensor(operator_list))        
    return loc_global_id, loc_sx_list, loc_sy_list, loc_sz_list

def spin_dephasing(op_list, size, gamma):
    """ 
    If a non-unitary Lindblad evolution is desired,
    this modeule constructs a list of collapse
    operators and collapse factors.
    By default, sigma-z collapse operators are chosen
    """
    loc_c_op_list = []; 
    loc_sz_list = op_list[3]
    collapse_weights = abs(gamma) * np.ones(size)
    loc_c_op_list = [np.sqrt(collapse_weights[n]) * loc_sz_list[n] for n in range(size)]
    return loc_c_op_list

# In [2]: 

def all_upto_two_body_spin_ops(op_list, size):
    """
    This module constructs all pair-wise combinations
    (ie. correlators) of trivial and non-trivial 
    one-body operators (ie. sx, sy, sz operators only)
    Note that there are N(N+1)/2 different correlators  
    in an N-site spin chain.
    """
    loc_global_id_list, sx_list, sy_list, sz_list = op_list

    pauli_four_vec = [loc_global_id_list, sx_list, sy_list, sz_list];
        
    sxsa_list = []; sysa_list = []; szsa_list = []; two_body_s = [];
    
    sxsa_list = [sx_list[n] * pauli_four_vec[a][b] for n in range(size)
                                                   for a in range(len(pauli_four_vec))
                                                   for b in range(len(pauli_four_vec[a]))]
    
    sysa_list = [sy_list[n] * pauli_four_vec[a][b] for n in range(size)
                                                   for a in range(len(pauli_four_vec))
                                                   for b in range(len(pauli_four_vec[a]))]
    
    szsa_list = [sz_list[n] * pauli_four_vec[a][b] for n in range(size)
                                                   for a in range(len(pauli_four_vec))
                                                   for b in range(len(pauli_four_vec[a]))]
    # Notice that since you have added the identity, this contains also 0-body (the global id)
    # and 1-body operators...
    two_body_s = [sxsa_list, sysa_list, szsa_list]
    return two_body_s

def two_body_spin_ops(op_list, size, build_all = False):
    """
    This module is redundant in its current form. 
    It basically either constructs all two-body 
    correlators or some subset of these. 
    """
    loc_list = []
    if build_all:
        loc_list = all_two_body_spin_ops(op_list, N)
    else: 
        globalid_list, sx_list, sy_list, sz_list = op_list       
        loc_sxsx = []; loc_sysy = []; loc_szsz = [];
        
        loc_sxsx = [sx_list[n] * sx_list[m] for n in range(size)
                                            for m in range(size)]
        loc_sysy = [sy_list[n] * sy_list[m] for n in range(size)
                                            for m in range(size)]
        loc_szsz = [sz_list[n] * sz_list[m] for n in range(size)
                                            for m in range(size)]
        loc_list.append(loc_sxsx)
        loc_list.append(loc_sysy)
        loc_list.append(loc_szsz)
    return loc_list

# In [3]: 

def Heisenberg_Hamiltonian(op_list, chain_type, size, Hamiltonian_paras, closed_bcs = True, visualization = False):
    """
    This module constructs the Heisenberg Hamiltonian for
    different types of systems, according to some 
    user-inputed parameters. 
    """
    spin_chain_type = ["XX", "XYZ", "XXZ", "XXX", "Anderson"]
    loc_globalid_list, sx_list, sy_list, sz_list = op_list       
          
    H = 0; N = size
    Jx = Hamiltonian_paras[0] * 2 * np.pi #* np.ones(N)
    h =  Hamiltonian_paras[3] * 2 * np.pi #* np.ones(N)
    H += sum(-.5* h * sz_list[n] for n in range(N-1)) # Zeeman interaction 
    
    if (chain_type in spin_chain_type): 
        if (chain_type == "XX"):
            H += sum(-.5* Jx *(sx_list[n]*sx_list[n+1] 
                                 + sy_list[n]*sy_list[n+1]) for n in range(N-1))
            if closed_bcs and N>2: 
                H -= .5* Jx *(sx_list[N-1]*sx_list[0] + sy_list[N-1]*sy_list[0])
            
        elif (chain_type == "XXX"):
            H += sum(-.5* Jx * (sx_list[n]*sx_list[n+1] 
                                 + sy_list[n]*sy_list[n+1]
                                 + sz_list[n]*sz_list[n+1]) for n in range(N-1))
            if closed_bcs and N>2: 
                H += -.5* Jx * (sx_list[N-1]*sx_list[0] 
                                 + sy_list[N-1]*sy_list[0]
                                 + sz_list[N-1]*sz_list[0])
        
        elif (chain_type == "XXZ"):
            Jz =  Hamiltonian_paras[2] * 2 * np.pi #* np.ones(N)
            H += sum(-.5 * Jx * (sx_list[n] * sx_list[n+1] + sy_list[n] * sy_list[n+1]) 
                     -.5 * Jz * (sz_list[n] * sz_list[n+1]) for n in range(N-1))
            if closed_bcs and N>2: 
                H += -.5 * Jx * (sx_list[N-1] * sx_list[0] +
                                 sy_list[N-1] * sy_list[0]) 
                -.5 * Jz * (sz_list[N-1] * sz_list[0])
        
        elif (chain_type == "XYZ"):
            Jy = Hamiltonian_paras[1] * 2 * np.pi 
            Jz = Hamiltonian_paras[2] * 2 * np.pi 
            H += sum(-.5 * Jx * (sx_list[n] * sx_list[n+1]) 
                     -.5 * Jy * (sy_list[n] * sy_list[n+1]) 
                     -.5 * Jz * (sz_list[n] * sz_list[n+1]) for n in range(N-1))
            if closed_bcs and N>2: 
                H += -.5 * Jx * (sx_list[N-1] * sx_list[0])
                -.5 * Jy * (sy_list[N-1] * sy_list[0]) 
                -.5 * Jz * (sz_list[N-1] * sz_list[0])
                
        elif (chain_type == "Anderson"):
            pass
    else:
        sys.exit("Currently not supported chain type")
              
    if visualization:
        qutip.hinton(H)
    assert mat_ansys.non_hermitianess_measure(H) < 1e-10, "Non-hermitian Hamiltonian obtained" 
    return H

def Heisenberg_Hamiltonian_tests(spin_ops_list, N):
    start_time = time.time()
    Hamiltonian_paras = [.2, .15, .1, 1.]
    spin_chain_type = ["XX", "XYZ", "XXZ", "XXX"]
    all_hamiltonians_are_hermitian = [False for i in range(2* len(spin_chain_type))]
    
    for i in range(len(spin_chain_type)):
        all_hamiltonians_are_hermitian[i] = qutip.isherm(Heisenberg_Hamiltonian(spin_ops_list, spin_chain_type[i],
                                                                              N, False, Hamiltonian_paras, False))
        if (all_hamiltonians_are_hermitian[i] == True):
            pass
        else:
            print(spin_chain_type[i], "Hamiltonian with open bcs non-hermitian")
                
    for i in range(len(spin_chain_type)):
        all_hamiltonians_are_hermitian[4+i] = qutip.isherm(Heisenberg_Hamiltonian(spin_ops_list, spin_chain_type[i],
                                                                              N, False, Hamiltonian_paras, True))
        
        if (all_hamiltonians_are_hermitian[i] == True):
            pass
        else:
            print(spin_chain_type[i], "Hamiltonian with closed bcs non-hermitian")
    
    if (Heisenberg_Hamiltonian_tests(spin_ops_list, N) == [True for i in range(2*len(spin_chain_type))]):
        print("All Hamiltonians are correct")
    print("--- Test concluded in: %s seconds ---" % (time.time() - start_time))
    return all_hamiltonians_are_hermitian

# In [4]:
        
def classical_ops(Hamiltonian, N, op_list, centered_x_op = False):
    
    identity_op = op_list[0][0]; sz_list = op_list[3]    
    labels = ["identity_op", "x_op", "p_op", "n_oc_op", "comm_xp", "corr_xp", "p_dot", "n_oc_disp"]
    
    cl_ops = {"identity_op": identity_op}
    if centered_x_op:
        cl_ops["x_op"] = sum((.5 + sz_list[k])*(k+1) for k in range(len(sz_list)))
    else:
        cl_ops["x_op"] = sum((k-N/2)*(sz_list[k] + .5 * identity_op) for k in range(len(sz_list)-1)) 
        
    cl_ops["p_op"] = 1j * mat_ansys.commutator(cl_ops["x_op"], Hamiltonian)
    cl_ops["n_oc_op"] = sum([sz_list[k] + .5 * identity_op for k in range(len(sz_list)-1)])
    cl_ops["comm_xp"] = .5 * mat_ansys.anticommutator(cl_ops["x_op"], cl_ops["p_op"])
    cl_ops["corr_xp"] = -1j * mat_ansys.commutator(cl_ops["x_op"], cl_ops["p_op"])
    cl_ops["p_dot"] = 1j * mat_ansys.commutator(Hamiltonian, cl_ops["p_op"])
    cl_ops["n_oc_disp"] = (cl_ops["n_oc_op"]-1.)**2
    
    for i in range(len(labels)):
        if qutip.isherm(cl_ops[labels[i]]):
            pass
        else:
            print(labels[i], "not hermitian")
    return cl_ops, labels
