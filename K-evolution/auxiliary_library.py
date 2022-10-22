# In [1]:

import qutip, sys, pickle
import numpy as np
import scipy.optimize as opt 
import matplotlib.pyplot as plt
import time as time
import scipy.linalg as linalg

# In [2]:

### This module checks if the matrix is positive definite ie. if all its eigenvalues are positive

def ev_checks(rho):
    if isinstance(rho, qutip.Qobj):
        rho = rho.full()
    try:
        np.linalg.cholesky(rho)
    except:
        return False
    return True


### This module checks if the user-input quantum object, rho, is a density operator or not.
### This is done by checking if it is a hermitian, positive definite, trace-one, matrix.
### Due to numerical instabilities, it may be possible that the trace is not exactly one, even though it is supposed to be,
### Therefore, a cut-off is implemented to determine if rho is, at least trace-wise, a matrix operator. 

def is_density_op(rho, verbose=False, critial=False):
    if not qutip.isherm(rho):
        if verbose:
            print("rho is not hermitician")
        assert not critical
        return False
    if abs(1 - rho.tr()) > 10**-10:
        if verbose:
            print("Tr rho !=1")
        assert not critical
        return False
    if not ev_checks(rho):
        if verbose:
            print("rho is not positive")
        assert not critical
        return False
    return True

def non_hermitianess_measure(rho):
    return linalg.norm(rho - rho.dag())

def null_matrix_check(rho):
    return (linalg.norm(rho) < 10**-10)

def commutator(A, B):
    result = 0
    if A.dims[0][0] == B.dims[0][0]: 
        pass
    else:
        raise Exception("Incompatible Qobj dimensions")
    result += A*B-B*A
    return result

def anticommutator(A, B):
    result = 0
    if A.dims[0][0] == B.dims[0][0]: 
        pass
    else:
        raise Exception("Incompatible Qobj dimensions")
    result += A*B+B*A
    return result

def Hamiltonian_comm_check(Hamiltonian, basis, labels = None, remove_null = True):
    if type(basis) is dict:
        for i in basis.copy(): 
            print("[H, ", i, "] = 0?: ", null_matrix_check(commutator(Hamiltonian, basis[i])))
            if remove_null and null_matrix_check(commutator(Hamiltonian, basis[i])):
                del basis[i]
                print(i, "basis element deleted")
    if type(basis) is list:
        for i in range(len(basis)):
            print("[H, ", labels[i], "] = 0?: ", null_matrix_check(commutator(Hamiltonian, basis[i])))
            if remove_null and null_matrix_check(commutator(Hamiltonian, basis[key])):
                basis.pop()
    return basis

def basis_hermitian_check(basis):
    if type(basis) is dict:
        basis = [basis[key] for key in basis]
    a = False
    for i in range(len(basis)):
        a = qutip.isherm(basis[i])
    return a

# In [3]: 

### Given an N-site spin chain, there are then 3N different, non-trivial, operators acting on the full Hilbert space.
### N sigmax operators, N sigmay operators, N sigmaz operators, and a global identity operator. 
### All these 3N+1-operators are constructed with a tensor product so that they all act on the full Hilbert space. 
### All, but the global identity operator, act non-trivially only on one Hilbert subspace. 

def one_body_spin_ops(size):
    
    ### Basic, one-site spin operators are constructed.
    
    loc_sx_list = []; loc_sy_list = []; loc_sz_list = []; loc_globalid_list = []
    id2 = qutip.qeye(2)
    sx = .5*qutip.sigmax()
    sy = .5*qutip.sigmay()
    sz = .5*qutip.sigmaz()
    
    ### The global identity operator is constructed 
    
    loc_global_id = [qutip.tensor([qutip.qeye(2) for k in range(size)])]
    
    ### Lists of one-body operators are constructed, so that they all act on the full Hilbert space. This is done
    ### via taking tensor products on lists of operators. 
    
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

### This module is relevant only if a non-unitary Lindblad evolution is chosen, it construcf a list of 
### collapse operators, with its corresponding collapse factors. In particular, sz collapse operators are chosen. 

def spin_dephasing(op_list, size, gamma):
        loc_c_op_list = []; 
        loc_sz_list = op_list[3]
        collapse_weights = abs(gamma) * np.ones(size)
        loc_c_op_list = [np.sqrt(collapse_weights[n]) * loc_sz_list[n] for n in range(size)]
        return loc_c_op_list

# In [4]: 

### This module constructs all pair-wise combinations (ie. correlators) of non-trivial one-body operators (ie. sx, sy, sz operators only). 
### There are N(N+1)/2 different correlators in an N-site spin chain.

def all_two_body_spin_ops(op_list, size):
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
    
    two_body_s = [sxsa_list, sysa_list, szsa_list]
    return two_body_s

# In [5]: 

### This module is redundant in its current form. It basically either constructs all two-body correlators 
### or some subset of these. 

def two_body_spin_ops(op_list, size, build_all = False):
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

# In [6]: 

### This module constructs the Heisenberg Hamiltonian for different types of systems, according to some user-inputed parameters. 

def Heisenberg_Hamiltonian(op_list, chain_type, size, Hamiltonian_paras, closed_bcs = True, visualization = False):
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
            Jy = Hamiltonian_paras[1] * 2 * np.pi #* np.ones(N)
            Jz = Hamiltonian_paras[2] * 2 * np.pi #* np.ones(N)
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
        
    if (qutip.isherm(H)): 
        return H
    else:
        sys.exit("Non-Hermitian Hamiltonian obtained")

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

#In [7]:

def prod_basis(b1, b2):
    return [qutip.tensor(b,s) for b in b1 for s in b2]

def HS_inner_prod_t(op1, op2, rho0=None): ### previous name: HS_inner_prod(A, B, rho0 = None):
    if (op1.dims[0][0]==op2.dims[0][0]):    ### Formally, this is the correct Hilbert-Schmidt inner product
        pass                                ### It is a complex valued inner product on the space of all endomorphisms 
    else:                                   ### acting on the N-partite Hilbert space 
        raise Exception("Incompatible Qobj dimensions")
    
    if rho0 is None:
        rho0 = qutip.qeye(op1.dims[0])
        rho0 = rho0/rho0.tr()        
    else:
        assert is_density_op(rho0, verbose=True), "rho0 is not a density op"
        
    result = 0
    result += (rho0 * (op1.dag() * op2)).tr()    
    return result

def HS_inner_prod_r(op1, op2, rho0): ### This inner product is real valued, provided both op1 and op2 are hermitian
    if (op1.dims[0][0]==op2.dims[0][0]):    ### and is easier to compute when dealing with spin chains, as the operator themselves 
        pass                                ### can be written as tensor products of local operators. A global-trace is then a product 
    else:                                   ### of traces over local Hilbert spaces
        raise Exception("Incompatible Qobj dimensions")
    
    if rho0 is None:
        rho0 = qutip.qeye(op1.dims[0])
        rho0 = rho0/rho0.tr()
    else:
        assert is_density_op(rho0, verbose=True), "rho0 is not a density op"
        
    result = 0
    result += .5 * (rho0 * anticommutator(op1.dag(), op2)).tr()    
    return result

def HS_inner_norm(op, rho0, sc_prod): ### previous name: mod_HS_inner_norm
    return sc_prod(op, op, rho0)

def HS_normalize_op(op, rho0, sc_prod):
    op = op/sc_prod(op, op, rho0)
    return op

def HS_distance(rho, sigma, rho0, sc_prod):
    if rho.dims[0][0]==sigma.dims[0][0]:
        pass
    else:
        raise Exception("Incompatible Qobj dimensions")
    
    return sc_prod(rho, sigma, rho0)

def base_orth(ops, rho0, sc_prod, visualization = False, reinforce_reality=False):
    
    if isinstance(ops, dict):
        ops = [ops[key] for key in ops]
    if isinstance(ops[0], list):
        ops = [op for op1l in ops for op in op1l]
    dim = ops[0].dims[0][0]
    basis = []
    for i, op in enumerate(ops): 
        op_norm = np.sqrt(sc_prod(op, op, rho0))
        op = op / op_norm
        alpha = np.array([sc_prod(op2, op, rho0) for op2 in basis])
        if reinforce_reality:
            alpha = alpha.real
        if visualization:
            print(alpha)
        op_mod = op - sum([c*op2 for c, op2, in zip(alpha, basis)])
        op_norm = np.sqrt(sc_prod(op_mod,op_mod,rho0))
        if op_norm > 1.e-5:
            if visualization:
                print("*****************norm", op_norm)
            op_mod = op_mod/(op_norm)
            basis.append(op_mod)
        else:
            if visualization:
                print("*****************skip", op, " of norm", op_norm)
    return basis

# In [7]: 

natural = tuple('123456789')

def n_body_basis(op_list, gr, N):
    basis = []
    globalid_list, sx_list, sy_list, sz_list = op_list       
        
    if (isinstance(gr,int) and str(gr) in natural):
        try:
            if (gr == 1):
                basis = globalid_list + sx_list + sy_list + sz_list
            elif (gr > 1):
                basis = [op1*op2 for op1 in n_body_basis(op_list, gr-1, N) for op2 in n_body_basis(op_list, 1, N)]
        except Exception as ex:
            basis = None
            print(ex)
    return basis

def max_ent_basis(op_list, op_basis_order_is_two, N, rho0, sc_prod):
    if (op_basis_order_is_two):
        basis = base_orth(n_body_basis(op_list, 2, N), rho0, sc_prod, False)  ## two-body max ent basis
        a = "two"
    else: 
        lista_ampliada = []
        for i in range(len(n_body_basis(op_list, 1, N))):
            lista_ampliada.append(qutip.tensor(n_body_basis(op_list, N,1)[i], qutip.qeye(2)))
        basis = base_orth(lista_ampliada, rho0, sc_prod, False) ## one-body max-ent basis
        a = "one"
    print(a + "-body operator chosen")
    return basis

# In [8]:

def n_body_max_ent_state(op_list, gr, N, coeffs = list, build_all = True, visualization = False):
    K = 0; rho_loc = 0;
    loc_globalid = qutip.tensor([qutip.qeye(2) for k in range(N)]) 
    globalid_list, sx_list, sy_list, sz_list = op_list       
    pauli_vec = [sx_list, sy_list, sz_list];
    
    if (gr == 1):
        try:
            K += sum(coeffs[n][m] *  one_body_spin_ops(N)[n][m] 
                                    for n in range(len(one_body_spin_ops(N)))
                                    for m in range(len(one_body_spin_ops(N)[n]))
                   ) 
            K += 10**-6 * loc_globalid
        except Exception as exme1:
            print(exme1, "Max-Ent 1 Failure")
            raise exme1
    elif (gr == 2): 
        try:
            K += sum(coeffs[n][m] * two_body_spin_ops(op_list, N, build_all)[n][m] 
                    for n in range(len(two_body_spin_ops(op_list, N, build_all)))
                    for m in range(len(two_body_spin_ops(op_list, N, build_all)[n]))
                   )
            K += 10**-6 * loc_globalid
        except Exception as exme2:
            print(exme2, "Max-Ent 2 Failure")
            raise exme2
    else:
        print('gr must be either 1 or 2')
    
    rho_loc = K.expm()
    rho_loc = rho_loc/rho_loc.tr()
    
    assert is_density_op(rho_loc, verbose=True), "rho_loc is not a density operator"
        
    if visualization: 
        qutip.hinton(rho_loc)
    return rho_loc 

# In [9]: 

def initial_state(op_list, N = 1, gaussian = True, gr = 1, x = .5, coeffs = list, psi0 = qutip.Qobj,
                  build_all = False, visualization=False):
    
    loc_globalid = qutip.tensor([qutip.qeye(2) for k in range(N)]) 
    if gaussian: 
        rho0 = n_body_max_ent_state(op_list, gr, N, coeffs, build_all, False)
    else:
        if (qutip.isket(psi0)):
            rho0 = psi0 * psi0.dag()
            rho0 = x * rho0 + (1-x)*loc_globalid * x/N
            rho0 = rho0/rho0.tr()
        else:
            print("Psi0 must be a ket")
    
    assert is_density_op(rho0, verbose=visualization), "Output is not a density operador"
    
    if visualization:
            qutip.hinton(rho0)
    return rho0  

# In [10]: 

def choose_initial_state_type(op_list, N, build_all, x, gaussian, gr):
    
    if (gaussian and gr == 1):
        a = len(op_list)
        b = len(op_list[0])
        coeffs_me1_gr1 = 10**-2.5 * np.full((a,b), 1)
        rho0 = initial_state(op_list, N, True, 1, None, coeffs_me1_gr1, None, build_all, False)
        statement = "One-body Gaussian"
        
    elif(gaussian and gr == 2):
        a = len(all_two_body_spin_ops(op_list, N))
        b = len(all_two_body_spin_ops(op_list, N)[0])

        coeffs_me2_gr2 = 10**-3 * np.full((a,b),1.)
        rho0 = initial_state(op_list, N, True, 2, None, coeffs_me2_gr2, None, build_all, False)
        statement = "Two-body Gaussian"
             
    elif(not gaussian):
        psi1_list = []
        psi1_list.append(qutip.basis(2,0))
        for n in range(N-1):
            psi1_list.append(qutip.basis(2,1))

        psi0 = qutip.tensor(psi1_list)
        rho0 = initial_state(op_list, N, False, None, .5, None, psi0, build_all, False)
        statement = "Non Gaussian"
      
    if gaussian:
         print(statement + " initial state chosen")
            
    return rho0

# In [12]: 

def logM(rho, svd = True):
    """
    Evaluates the logarithm of a positive matrix rho
    """
    assert ev_checks(rho), "Non positive-defined input matrix"
    
    if isinstance(rho, qutip.Qobj):
        qutip_form = True
        dims = rho.dims
    else:
        qutip_form = False        

    if svd:            
        if qutip_form:
            rho = rho.full()
        U, Sigma, Vdag = linalg.svd(rho, full_matrices = False)
        matrix_log = U @ np.diag(np.log(Sigma)) @ U.conj().transpose() 
        #matrix_log = qutip.Qobj(U) * qutip.Qobj(np.diag(np.log(Sigma))) * qutip.Qobj(np.array(Vdag))
    else: 
        if qutip_form:
            eigvals, eigvecs = rho.eigenstates()
            matrix_log = sum([np.log(vl)*vc*vc.dag() for vl, vc in zip(eigvals, eigvecs)]) 
        else:
            rho = rho.full()
            eigvals, eigvecs = linalg.eigh(rho)
            return evecs @ np.array([np.log(ev)*np.array(f) for ev, f in zip(evals, evecs.transpose().conj())])
    
    if qutip_form:
        matrix_log = qutip.Qobj(matrix_log, dims)
    return matrix_log


def sqrtM(rho, svd = True):
    """
    Evaluates the square root of a positive matrix rho
    """
    assert ev_checks(rho), "Non positive-defined input matrix"
    
    if isinstance(rho, qutip.Qobj):
        qutip_form = True
        dims = rho.dims
    else:
        qutip_form = False

    if svd:
        if qutip_form:
            rho = rho.full()
        U, Sigma, Vdag = linalg.svd(rho, full_matrices = False)
        matrix_log = U @ np.diag((Sigma)**.5) @ U.conj().transpose() 
        #matrix_log = qutip.Qobj(U) * qutip.Qobj(np.diag(np.log(Sigma))) * qutip.Qobj(np.array(Vdag))
    else: 
        if qutip_form:
            eigvals, eigvecs = rho.eigenstates()
            matrix_log = sum([(vl**.5)*vc*vc.dag() for vl, vc in zip(eigvals, eigvecs)]) 
        else:
            rho = rho.full()
            eigvals, eigvecs = linalg.eigh(rho)
            return evecs @ np.array([(ev**.5)*np.array(f) for ev, f in zip(evals, evecs.transpose().conj())])
    
    if qutip_form:
        matrix_log = qutip.Qobj(matrix_log, dims)
    return matrix_log



def bures(rho, sigma, svd = True):
    """
    Evaluates the Bures metric between two states. 
    """
    assert is_density_op(rho), "rho is not a density operator"
    assert is_density_op(sigma), "sigma is not a density operator"
    
    sqrt_sigma = sqrtM(sigma.full(), svd=svd)
    fidelity = sqrtM((sqrt_sigma @ rho.full()  @sqrt_sigma),svd=True).trace().real

    assert abs(fidelity.imag)<1.e-10, f"complex fidelity? fidelity={fidelity}"
    fidelity = fidelity.real
    assert 0<=fidelity, f"negative fidelity? fidelity={fidelity}"
    if fidelity>1.:
        assert (fidelity-1)<1.e-8, f"error in fidelity too large fidelity={fidelity}"
        return 0.
    
    return  np.arccos(fidelity)/np.pi
    

# In [13]: 

def proj_op(K, basis, rho0, sc_prod):
    return sum([sc_prod(b, K,rho0) * b for b in basis])

def rel_entropy(rho, sigma, svd = True):
    assert  ev_checks(rho), "rho is not positive"
    assert  ev_checks(sigma), "sigma is not positive"
    val = (rho*(logM(rho, svd) - logM(sigma, svd))).tr()
    assert abs(val.imag)/(abs(val.real)+1e-8) < 1.e-3, f"imaginary part larger than the tolerance...val={val}"
    return val.real

# In [14]:
        
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
        basis = base_orth(basis, rho0, sc_prod, False)
    except:
        print("orth error")
        raise
    try:
        sigma = proj_op(logM(rho), basis, rho0, sc_prod).expm()
        sigma = (sigma+sigma.dag())/(2.*sigma.tr())
    except:
        print("gram error")
    try:
        return distance(rho, sigma)
    except:
        print("fail error proj state")
        return None
    
# In [15]:

def legacy_classical_ops(n, Hamiltonian):
    id_loc = qutip.qeye(2)
    sz_loc = .5*qutip.sigmaz()
    sx_loc = .5*qutip.sigmax()
    sy_loc = .5*qutip.sigmay()

    n_oc =  sum(qutip.tensor([id_loc for i in range(k)]+ 
                     [(sz_loc + .5*id_loc)]+ 
                     [id_loc for i in range(n-k-1)]
                    ) for k in range(n-1))
    x = sum(qutip.tensor([id_loc for i in range(k)]+ 
                     [(k-n/2)*(sz_loc + .5*id_loc)]+ 
                     [id_loc for i in range(n-k-1)]
                    ) for k in range(n-1))
    Mauricio_noc = sum([qutip.tensor([id_loc for i in range(k)]+ 
                     [(sz_loc + .5*id_loc)]+ 
                     [id_loc for i in range(n-k-1)]) for k in range(n-1)])
    Tom_noc = sum([spin_ops_list[3][k] + .5 * spin_ops_list[0][0] for k in range(n-1)])
    Mauriciox = sum(qutip.tensor([id_loc for i in range(k)]+ 
                     [(k-n/2)*(sz_loc + .5*id_loc)]+ 
                     [id_loc for i in range(n-k-1)]
                    ) for k in range(n-1))
    Tomix = sum((k-n/2)*(spin_ops_list[3][k] + .5 * spin_ops_list[0][0]) for k in range(n-1))
    return None

def classical_ops(Hamiltonian, N, op_list, centered_x_op = False):
    
    identity_op = op_list[0][0]; sz_list = op_list[3]    
    labels = ["identity_op", "x_op", "p_op", "n_oc_op", "comm_xp", "corr_xp", "p_dot", "n_oc_disp"]
    
    cl_ops = {"identity_op": identity_op}
    if centered_x_op:
        cl_ops["x_op"] = sum((.5 + sz_list[k])*(k+1) for k in range(len(sz_list)))
    else:
        cl_ops["x_op"] = sum((k-N/2)*(sz_list[k] + .5 * identity_op) for k in range(len(sz_list)-1)) 
        
    cl_ops["p_op"] = 1j * commutator(cl_ops["x_op"], Hamiltonian)
    cl_ops["n_oc_op"] = sum([sz_list[k] + .5 * identity_op for k in range(len(sz_list)-1)])
    cl_ops["comm_xp"] = .5 * anticommutator(cl_ops["x_op"], cl_ops["p_op"])
    cl_ops["corr_xp"] = -1j * commutator(cl_ops["x_op"], cl_ops["p_op"])
    cl_ops["p_dot"] = 1j * commutator(Hamiltonian, cl_ops["p_op"])
    cl_ops["n_oc_disp"] = (cl_ops["n_oc_op"]-1.)**2
    
    for i in range(len(labels)):
        if qutip.isherm(cl_ops[labels[i]]):
            pass
        else:
            print(labels[i], "not hermitian")
    return cl_ops, labels
    
HS_modified = True

class Result(object):
      def __init__(self, ts=None, states=None):
        self.ts = ts
        self.states = states
        self.projrho0_app = None   
        self.projrho_inst_app = None 

rhos = []    
def callback_A(t, rhot):
    global rho
    global rhos
    rho = rhot
    rhos.append(rhot)
    
def spin_chain_ev(size, init_state, chain_type, closed_bcs, Hamiltonian_paras, omega_1=3., omega_2=3., temp=1, tmax = 250, deltat = 10, 
                  two_body_basis = True, unitary_ev = False, gamma = 1*np.e**-2,
                  gaussian = True, gr = 2, xng = .5, sc_prod = HS_inner_prod_r, obs_basis = None, do_project = True):
    
    build_all = True
    
    def callback_t(t, rhot):
        global rho
        rho = rhot
    
    sc_prod = HS_inner_prod_r
    ### The algorithm starts by constructing all one-body spin operators, acting on the full N-particle Hilbert space
    ### This means, it constructs the 3N + 1 one_body spins ops (N sigmax operators, N sigmay operators, N sigmaz operators
    ### an the global identity operator
    
    spin_big_list = one_body_spin_ops(size)
    loc_globalid = one_body_spin_ops(size)[0][0]
    
    #Jx = Hamiltonian_paras[0]; Jy = Hamiltonian_paras[1]
    #Jz = Hamiltonian_paras[2]; h = Hamiltonian_paras[3] 
    
    ### Then, the algorithm either takes a user-input initial density matrix or it constructs a default one.
    
    if init_state is None:
        print("Processing default initial state")
        rho0 = choose_initial_state_type(spin_big_list, size, build_all, xng, gaussian, gr)
    else: 
        print("Processing custom initial state")
        if (is_density_op(init_state)):
            rho0 = init_state
        else:
            raise Exception("User input initial state not a density matrix")
    
    ### Hamiltonian
    
    H=Heisenberg_Hamiltonian(op_list = spin_big_list, chain_type = chain_type,
                                size = size, Hamiltonian_paras = Hamiltonian_paras,
                                closed_bcs = closed_bcs, visualization = False)
    
    ### Then, the algorithm either takes a user-input choice for observables or it constructs a default one. 
    
    if obs_basis is None: 
        print("Processing default observable basis")
        cl_ops, labels = classical_ops(H, size, spin_big_list, False)
        obs = cl_ops #, x_op**2,p_op**2, corr_op, p_dot]
    else:
        print("Processing custom observable basis")
        obs = obs_basis
        
    sampling = max(int(10*max(1,omega_1, omega_2)*deltat), 10)
    print("sampling:", sampling)
    
    ### If a unitary evolution is chosen, no colapse operators nor colapse factors are taken into account. 
    ### Otherwise, a default sz-colapse operator list is chosen. 
    
    if unitary_ev: 
        print("Closed evolution chosen")
        c_op_list = None
    else:
        print("Open evolution chosen")
        c_op_list = spin_dephasing(spin_big_list, size, gamma)
        
    rho = init_state                                                               
    approx_exp_vals = [[qutip.expect(op, rho) for op in obs]]
    ts= [0]
    
    ### If a projected evolution is desired, then a two-body spin operator basis is chosen. Otherwise, if the exact ev,
    ### is desired, this step will be skipped. 
    
    if do_project:    
        print("Processing two-body for proj ev")
        basis = max_ent_basis(spin_big_list, two_body_basis, size, rho0, sc_prod)
    
    for i in range(int(tmax/deltat)):
        ### Heisenberg Hamiltonian is constructed
        qutip.mesolve(H,
                               rho0=rho, 
                               tlist=np.linspace(0,deltat, sampling), 
                               c_ops=c_op_list, 
                               e_ops=callback_t,
                               args={'gamma': gamma,'omega_1': omega_1, 'omega_2': omega_2}
                               )
        ts.append(deltat*i)
        if do_project:
            rho = proj_op(logM(rho), basis, rho0, sc_prod)
            e0 = max(rho.eigenenergies())
            rho = rho - loc_globalid * e0
            rho = rho.expm()
            trrho = (2.*rho.tr())
            rho = (rho+rho.dag())/trrho
            rhos.append(rho)

        #print(qutip.entropy.entropy_vn(rho))
        newobs = [qutip.expect(rho, op) for op in obs]
        approx_exp_vals.append(newobs)
        
    result = {}
    result["ts"] = ts
    result["averages"] = np.array(approx_exp_vals)
    result["State ev"] = rhos
    
    if unitary_ev:
        title = f"{chain_type}-chain closed ev/Proj ev for N={size} spins" 
    else:
        title = f"{chain_type}-chain open ev/Proj ev for N={size} spins" 
    
    #with open(title+".pkl","wb") as f:
    #    pickle.dump(result, f)
    
    #print("type rho=", type(result["State ev"]))
    
    ev_parameters = {"no. spins": size, "chain type": chain_type, "Model parameters": Hamiltonian_paras, "Sampling": sampling, 
                     "Two body basis": two_body_basis, "Closed ev": unitary_ev, "Colapse parameters": gamma, 
                     "Gaussian ev": gaussian, "Gaussian order": gr, "Non-gaussian para": xng, "Type of inner product": sc_prod,
                     "no. observables returned": len(obs), "Proj. ev": do_project}
    
    return title, ev_parameters, result

# In [16]: 

def build_reference_state(size, temp, Hamiltonian, lagrange_op, lagrange_mult, svd = True):
    ### building the reference state
    k_B = 1; beta = 1/(k_B * temp); 
    K = -beta * (Hamiltonian - lagrange_mult * (lagrange_op - 1)**2)
    K = K - max(K.eigenenergies(K)) 
    rho_ref = K.expm()
    rho_ref = rho_ref/rho_ref.tr()
    if not is_density_op(rho_ref):
        if (not qutip.isherm(rho_ref) or non_hermitianess_measure(rho_ref) < 1e-10):
            # svd? 
            rho_ref = .5 * (rho_ref + rho_ref.dag())
        if not ev_checks(rho_ref):
            sys.exit("Singular density op")
            
    #if svd:     #    Ks_max_value = max(linalg.svd(K)[1])
    #if not svd:    #    Ks_max_value = min(linalg.eigvals(K).real)
    return K, rho_ref

def recursive_basis(depth, Hamiltonian, seed_op, rho0): 
    basis = [seed_op]; loc_op = 0
    if depth > 0: 
        for i in range(1, depth):
            loc_op = qutip.Qobj(-1j * commutator(Hamiltonian, basis[i-1]))
            if (linalg.norm(loc_op) < 1e-10):
                print("Operator at depth", i, "is null")
                loc_op = None
                break
            loc_op = (loc_op * rho0).tr() - loc_op
            basis.append(loc_op)
    elif (depth == 0):
        basis = []
    return basis

def vectorized_recursive_basis(depth_and_ops, Hamiltonian, rho0):        
    basis_rec = []
    for depth, op in depth_and_ops: 
        basis_rec += recursive_basis(depth, Hamiltonian, op, rho0)
    return basis_rec
    
# In [17]:

def H_ij_matrix(Hamiltonian, basis, rho0, sc_prod):
    coeffs_matrix = np.array([[sc_prod(op1, -1j * commutator(Hamiltonian, op2), rho0) for op2 in basis] for op1 in basis])
    return coeffs_matrix

def basis_orthonormality_check(basis, rho0, sc_prod): 

    dim = len(basis)
    
    hermitian_basis = [linalg.norm(op1 - op1.dag()) < 1e-10 for op1 in basis]
    assert np.all(hermitian_basis), ("Not all the operators are "
                                     f"hermitician:\n {hermitian_basis}")

    null_averages = [np.real((rho0 * op1).tr()) < 1e-10 for op1 in basis]
    assert all(null_averages[1:]), ("Some operators do not have a null average:\n" 
                                    f"{null_averages}")
    
    gram_matrix = [[sc_prod(op2, op1, rho0) for op2 in basis] for op1 in basis]
    
    normalized = [abs(gram_matrix[i][i]-1.)<1e-10  for i in range(dim)]
    assert all(normalized), ("Some operators in the basis are not normalized:\n"
                             f"{normalized}")

    assert (linalg.norm(
        (np.identity(dim) - gram_matrix) < 10**-10)), "Not all operators are pair-wise orthogonal"
    
    print("The basis is orthonormal and hermitian")
    return True

# In [18]:

def build_rho0_from_basis(basis, temp=1.):
    phi0 = [np.random.rand()/temp for b in basis]
    k0 = -sum( f*op for f,op in zip(phi0, basis))
    k0 = k0 - max(k0.eigenenergies())
    rho0 = (k0).expm()
    normalizacion = rho0.tr()
    rho0 = rho0 / normalizacion
    assert is_density_op(rho0, verbose=True), "rho is not a density matrix."
    return phi0, rho0

def visz_H_tensor_evs(Htensor):
    if (type(Htensor) == qutip.Qobj):
        Htensor_local = np.array(Htensor)
    x = sorted(np.array(qutip.Qobj(Htensor_local).eigenenergies().real))
    y = sorted(np.array(qutip.Qobj(Htensor_local).eigenenergies().imag))
    z = np.arange(len(x))
    fig1, ax1 = plt.subplots()
    ax1.plot(z,x, label = "Real Part evs")
    ax1.plot(z,y, label = "Imag part evs")
    ax1.legend(loc=0)
    ax1.set_title("H-tensor's eigenvalues' real and imag part")

def semigroup_phit_and_rhot_sol(phi0, rho0, Htensor, ts, basis):
    Phi_vector_solution = []; rho_at_timet = []
    phi0=np.array(phi0)
    Phi_vector_solution.append(phi0); rho_at_timet.append(rho0)
    
    new_phi = Phi_vector_solution[0]
    for i in range(1, len(ts)-1):
        evol_op = linalg.expm(ts[i]*Htensor)
        new_phi = evol_op.dot(phi0)
        Phi_vector_solution.append(new_phi)
        K = -sum( f*op for f,op in zip(new_phi, basis))
        K = K - max(K.eigenenergies()) 
        # assert K.isherm, "K is not Hermitician "
        if not K.isherm:
            print("Non hermitician part norm:", np.linalg.norm( (K-K.dag()).full())  )
            assert K.isherm, "K is not Hermitician "
        rhot= K.expm()
        #if (rhot.tr() < 1e-6):
        #   continue 
        rho_at_timet.append(rhot/rhot.tr())
        #rho_at_timet.append(rhot)
    return rho_at_timet, Phi_vector_solution    

def semigroup_rhos_test(rho_list, visualization_nonherm, ts):
    non_densitiness = [ (linalg.norm(rho_list[t] - rho_list[t].dag())/ linalg.norm(rho_list[t])) for t in range(len(rho_list))]
    rho_list = [.5 * (rho_list[t] + rho_list[t].dag()) for t in range(len(rho_list))]
            
    if visualization_nonherm:
        x2 = np.arange(len(non_densitiness))
        y2 = non_densitiness
        fig2, ax2 = plt.subplots()
        ax2.plot(x2,y2)
        ax2.legend(loc=0)
        ax2.set_title("Non-hermitian measure for semigroup states")
    return rho_list

def LEGACY_plots(ts, res_proj_ev, res_exact):
    z = ts[:-1]
    fig3, ax3 = plt.subplots()
    ax3.plot(z, res_proj_ev[0], label = "Manifold-proj")
    ax3.plot(z, res_exact.expect[0][:-1], label = "Exact")
    ax3.legend(loc=0)
    ax3.set_title("Expected values for x_op - Exact v. Proj. ev. ")
        
    fig4, ax4 = plt.subplots()
    ax4.plot(z, res_proj_ev[1], label = "Manifold-proj")
    ax4.plot(z, res_exact.expect[1][:-1], label = "Exact")
    ax4.legend(loc = 0)
    ax4.set_title("Expected values for n_oc_op - Exact v. Proj. ev.")
        
    fig5, ax5 = plt.subplots()
    ax5.plot(z, res_proj_ev[2], label = "Manifold-proj")
    ax5.plot(z, res_exact.expect[2][:-1], label = "Exact")
    ax5.legend(loc = 0)
    ax5.set_title("Expected values for magnetization - Exact v. Proj. ev.")
    
def plot_exact_v_proj_ev_avgs(observables, label, ts, res_proj_ev, res_exact):
    Tot = len(observables); Cols = 2
    Rows = Tot // Cols 
    if Tot % Cols != 0:
        Rows += 1
    Position = range(1,Tot + 1)
    z = ts[:-1]
    fig = plt.figure(figsize=(16, 32))
    for k in range(Tot):
        ax = fig.add_subplot(Rows,Cols,Position[k])
        ax.plot(z, res_exact.expect[k][:-1], label = "Exact")
        ax.plot(z, res_proj_ev[k], label = "Manifold proj")        
        ax.legend(loc=0)
        ax.set_title("Expected values: Proj-ev. v. Exact for " + label[k])
    plt.show()

    
    
def mesolve(H, rho0, tlist, c_ops=None, e_ops=None,**kwargs):
    """
    Wrapper for the qutip.mesolve function that allows to get
    both the expectation values and the operators.
    """
    from typing import Callable
    if e_ops is None or isinstance(e_ops, Callable):
        return qutip.mesolve(H, rho0, tlist, c_ops, e_ops=e_ops, **kwargs)
    
    result = qutip.solver.Result()
    result.expect = [[] for e in e_ops]
    result.times = []

    def callback(t, rho):
        result.times.append(t)
        result.states.append(rho)
        for i, e in enumerate(e_ops):
            result.expect[i].append(qutip.expect(rho, e))

    qutip.mesolve(H, rho0, tlist, c_ops, e_ops=callback, **kwargs)
    return result
