# In [0]:

import qutip
import numpy as np
import scipy.linalg as linalg 

# In [1]:

def one_body_spin_ops(args):
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
    
    size=args['size']; tenszd_spin_ops={}
    tenszd_spin_ops['sx'] = []; tenszd_spin_ops['sy'] = []; tenszd_spin_ops['sz'] = []
    
    id2 = qutip.qeye(2)
    sx = .5*qutip.sigmax(); sy = .5*qutip.sigmay(); sz = .5*qutip.sigmaz()
    
    ### The global identity operator is constructed 
    tenszd_spin_ops['idop'] = qutip.tensor([id2 for k in range(size)]) 
    
    ### Tensorized-lists of one body spin operators
    
    for n in range(size):
        operator_list = []
        for m in range(size):
            operator_list.append(id2)
        operator_list[n] = sx
        tenszd_spin_ops['sx'].append(qutip.tensor(operator_list))
        
        operator_list[n] = sy
        tenszd_spin_ops['sy'].append(qutip.tensor(operator_list))
        
        operator_list[n] = sz
        tenszd_spin_ops['sz'].append(qutip.tensor(operator_list))        
        
    return tenszd_spin_ops

# In [2]: 

def Heisenberg_1D_Hamiltonian(args, spin_ops, closed_bcs=True, visualization=False, tol=1e-10):
    """
    This module constructs different types of nearest-neighbour
    Heisenberg-like spin chains, using a list of local [see Warnings and Remarks] spin operators.
    This module takes as input the following parameters:
    
        *♥*♥* 1. spin_ops: a list of hermitian local [see Warnings and Remarks further below] spin operators.
        *♥*♥* 2. closed_bcs: boolean parameter, default value = True. 
                             If a spin chain has more than two lattice sites, the spin chain can be made
                             translationally invariant by adding an extra term corresponding to the N-th site's interaction
                             with the first site. 
        *♥*♥* 3. visualization: boolean parameter, default value = False.
                                Toggling this option will output a Hinton diagram of the Hamiltonian 
                                
        ====> Returns: the system's Hamiltonian.
        Warnings and Remarks: a. by "local spin operators", it is understood that these spin operators act on the global
                                 Hilbert space, having already been written in terms of tensor products. These spin operators
                                 act non-trivially only on a single lattice site, and act as an identity operator otherwise.
                              b. If a spin chain is chosen to be, eg., an XX chain, only the 
                                 first and fourth parameters will be use, the others will be ignored.
        
    """          
    N = args['size']; chain_type = args['chain_type']
    
    H = 0
    try:
        H += sum(args['h'] * spin_ops['sz'][n]for n in range(N-1)) # Zeeman interaction=
    except:
        H += 0
    
    if (chain_type == "XX"):
        H += sum(args['Jx'] *(spin_ops['sx'][n]*spin_ops['sx'][n+1] 
                                   +spin_ops['sy'][n]*spin_ops['sy'][n+1]) for n in range(N-1))
        if closed_bcs and N>2: 
            H +=args['Jx']*(spin_ops['sx'][N-1]*spin_ops['sx'][0] + spin_ops['sy'][N-1]*spin_ops['sy'][0])
        
    elif (chain_type == "XY"):
        H += sum(args['Jx']*spin_ops['sx'][n]*spin_ops['sx'][n+1] 
                       +args['Jy']*spin_ops['sy'][n]*spin_ops['sy'][n+1] for n in range(N-1))
            
        if closed_bcs and N>2:
            H += (args['Jx']* spin_ops['sx'][N-1]*spin_ops['sx'][0] 
                 +args['Jy'] * spin_ops['sy'][N-1]*spin_ops['sy'][0])
            
    elif (chain_type == "XXX"):
        H += sum(args['Jx'] * (spin_ops['sx'][n]*spin_ops['sx'][n+1] 
                                     + spin_ops['sy'][n]*spin_ops['sy'][n+1]
                                     + spin_ops['sz'][n]*spin_ops['sz'][n+1]) for n in range(N-1))
        if closed_bcs and N>2: 
            H += args['Jx'] * (spin_ops['sx'][N-1]*spin_ops['sx'][0] 
                                     + spin_ops['sy'][N-1]*spin_ops['sy'][0]
                                     + spin_ops['sz'][N-1]*spin_ops['sz'][0])
        
    elif (chain_type == "XXZ"):
        H += sum(-.5 * args['Jx'] * (spin_ops['sx'][n] * spin_ops['sx'][n+1] + spin_ops['sy'][n] * spin_ops['sy'][n+1]) 
                     -.5 * args['Jz'] * (spin_ops['sz'][n] * spin_ops['sz'][n+1]) for n in range(N-1))
        if closed_bcs and N>2: 
            H += (-.5 * args['Jx'] * (spin_ops['sx'][N-1] * spin_ops['sx'][0] +
                                     spin_ops['sy'][N-1] * spin_ops['sy'][0]) 
                 -.5 * args['Jz'] * (spin_ops['sz'][N-1] * spin_ops['sz'][0]))
        
    elif (chain_type == "XYZ"):
        H += sum(-.5 * args['Jx'] * (spin_ops['sx'][n] * spin_ops['sx'][n+1]) 
                     -.5 * args['Jy'] * (spin_ops['sy'][n] * spin_ops['sy'][n+1]) 
                     -.5 * args['Jz'] * (spin_ops['sz'][n] * spin_ops['sz'][n+1]) for n in range(N-1))
        if closed_bcs and N>2: 
            H += (-.5 * args['Jx'] * (spin_ops['sx'][N-1] * spin_ops['sx'][0])
                  -.5 * args['Jy'] * (spin_ops['sy'][N-1] * spin_ops['sy'][0]) 
                  -.5 * args['Jz'] * (spin_ops['sz'][N-1] * spin_ops['sz'][0]))
                
    else:
        sys.exit("Currently not supported chain type")
              
    if visualization:
        qutip.hinton(H)
    assert linalg.norm(H - H.dag()) < tol, "Non-hermitian output obtained" 
    return H
