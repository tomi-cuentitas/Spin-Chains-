# In [1]:

import qutip, sys, pickle
import numpy as np
import scipy.optimize as opt 
import matplotlib.pyplot as plt
import time as time
import scipy.linalg as linalg

import a_matrix_analysis_lib as mat_ansys
import b_spin_representations as su2
import c_evs_visualization_tools as evs_plot

# In [1]:

def build_reference_state(temp, Hamiltonian, lagrange_op, lagrange_mult):
    """
    This module constructs a gaussian reference 
    state, to then be used by the Hilbert-Schmidt 
    inner-products. It takes as parameters
    
    ***. a temperature, 
    ***. a Hamiltonian,
    ***. a custom hermitian operator with
    ***. an associated real-valued weight.
    
    It returns a gaussian state
    """
    ### building the reference state
    k_B = 1; beta = 1/(k_B * temp); 
    K = -beta * (Hamiltonian - lagrange_mult * (lagrange_op - 1)**2)
    K = K - max(K.eigenenergies(K)) 
    rho_ref = K.expm()
    rho_ref = rho_ref/rho_ref.tr()
    if not mat_ansys.is_density_op(rho_ref):
        if (mat_ansys.non_hermitianess_measure(rho_ref) <= 1e-10):
            rho_ref = .5 * (rho_ref + rho_ref.dag())
        assert mat_ansys.ev_checks(rho_ref), "Singular density op"
    return K, rho_ref
    
# In [2]:

def H_ij_matrix(Hamiltonian, basis, rhoref, sc_prod):
    coeffs_matrix = np.array([[sc_prod(op1, -1j * mat_ansys.commutator(Hamiltonian, op2), rhoref) for op2 in basis] for op1 in basis])
    return coeffs_matrix

def basis_orthonormality_check(basis, rhoref, sc_prod, visualization_Gram_m = False,
                                                     title_format_dprojev = True): 
    
    if rhoref is None:
        rhoref = qutip.qeye(basis[0].dims[0])
        rhoref = rhoref/rhoref.tr()
    
    dim = len(basis)
    hermitian_basis = [mat_ansys.non_hermitianess_measure(op1) <= 1e-5 for op1 in basis]
    assert np.all(hermitian_basis), ("Not all the operators are "
                                     f"hermitian:\n {hermitian_basis}")
    
    gram_matrix = [[sc_prod(op2, op1, rhoref) for op2 in basis] for op1 in basis]
    normalized = [abs(gram_matrix[i][i]-1.) <= 1e-5  for i in range(dim)]
    assert all(normalized), ("Some operators in the basis are not normalized:\n"
                             f"{normalized}")

    assert (linalg.norm((np.identity(dim) - gram_matrix) <= 1e-10)), "Not all operators are pair-wise orthogonal"
    
    if title_format_dprojev:
        print("          -. Check passed: the basis is orthonormal and hermitian")
    else: 
        print("The basis is orthonormal and hermitian")

    null_averages = [np.real((rhoref * op1).tr()) <= 1e-6 for op1 in basis]
    assert all(null_averages[1:]), ("Some operators do not have a null average:\n" 
                                    f"{null_averages}")
    
    gram_matrix = qutip.Qobj(gram_matrix)
    if visualization_Gram_m:
        gram_eigen = gram_matrix.eigenenergies()
        plt.scatter([i+1 for i in range(len(gram_eigen))], gram_eigen, 
        label = "Gram matrix eigenvalues")
    return gram_matrix

# In [3]:

def build_rho0_from_basis(coeff_list, basis, temp): 
    """
    This module, using an operator basis and an initial
    array of numbers, constructs the gaussian initial 
    state 
    """
    beta = 1/temp
    if (coeff_list == None):
        coeff_list = [0.] + [np.random.rand() for i in range(len(basis) - 1)]
     
    loc_coeff_list = coeff_list
    rho0 = (-sum( f*op  for f, op in zip(loc_coeff_list, basis))).expm()
    rho0 = rho0/rho0.tr()
    #loc_coeff_list[0] = np.log(rho0.tr())
    #rho0 = (-sum( f*op  for f, op in zip(loc_coeff_list, basis))).expm()
    
    assert mat_ansys.is_density_op(rho0, verbose=True), "rho is not a density matrix."
    return loc_coeff_list, rho0

def semigroup_phit_and_rhot_sol(phi0, rho0, Htensor, ts, basis):
    """
    This module constructs the solution to the differential equation:
    
    dphi(t)/dt = Htensor * phi(t),
    
    where Htensor is a real-valued N x N matrix. From this phit-list,
    the rhot-list is constructed.
    
    This module takes as parameters:
    
    *. an initial configuration of parameters for the chosen basis, phi0,
    *. an initial density matrix,
    *. an H-tensor, constructed from the basis,
    *. a list of times,
    *. and the basis of operators.
    """
    
    Phi_vector_solution = []; rho_at_timet = []; tracerho_at_timet = []
    phi0=np.array(phi0)
    Phi_vector_solution.append(phi0); rho_at_timet.append(rho0)
    
    new_phi = Phi_vector_solution[0]
    tracerho_at_timet = [0]
    for i in range(1, len(ts)-1):
        evol_op = linalg.expm(ts[i]*Htensor)
        new_phi = evol_op.dot(phi0)
        Phi_vector_solution.append(new_phi)
        K = -sum( f*op for f,op in zip(new_phi, basis))
        K = K - max(K.eigenenergies()) 
        if not (np.linalg.norm( (K-K.dag()).full()) < 1e-5):
            print("Non hermitician part norm:", np.linalg.norm( (K-K.dag()).full())  )
            assert K.isherm, "K is not Hermitician "
        rhot= K.expm()
        tracerho_at_timet.append(rhot.tr())
        rho_at_timet.append(rhot/rhot.tr())
    return rho_at_timet, Phi_vector_solution, tracerho_at_timet 

def semigroup_rhos_test(rho_list, visualization_nonherm, ts):
    non_densitiness = [ (mat_ansys.non_hermitianess_measure(rho_list[t])/linalg.norm(rho_list[t])) for t in range(len(rho_list))]
    rho_list = [.5 * (rho_list[t] + rho_list[t].dag()) for t in range(len(rho_list))]
            
    if visualization_nonherm:
        x2 = np.arange(len(non_densitiness))
        y2 = non_densitiness
        fig2, ax2 = plt.subplots()
        ax2.plot(x2,y2)
        ax2.legend(loc=0)
        ax2.set_title("Non-hermitian measure for semigroup states")
    return rho_list

# In [13]:
    
def exact_v_proj_ev_matrix_metrics(ts, res_proj_ev_rhot_list, res_exact):    
    ts_prime = ts[:-1]
    bures_exact_v_proj_ev_list = mat_ansys.bures_vectorized(rhot_list = res_proj_ev_rhot_list,
                                                  sigmat_list = res_exact.states[:-1])
    
    relent_ex_v_proj_ev_list, relent_proj_ev_v_ex = mat_ansys.relative_entropies_vectorized(rhot_list = res_proj_ev_rhot_list,
                                                                                  sigmat_list = res_exact.states[:-1])
    
    return bures_exact_v_proj_ev_list, relent_ex_v_proj_ev_list, relent_proj_ev_v_ex
    
def mod_mesolve(H, rho0, tlist, c_ops=None, e_ops=None,**kwargs):
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

# In [14]:

def max_ent_ev(timespan, Hamiltonian, depth_and_seed_ops, 
               observables, label_ops, 
               temp_ref, temp_rho,
               rho0, K0, custom_ref_state, custom_Kref, coeff_list, 
               reference_state_choice = None,
               build_referential_thermal_state = False, rho_ref_equal_rho0 = False, 
               compute_exact_dynamics = True,
               visualize_H_evs = False, visualization_nonherm = False, visualize_expt_vals = True, visualize_diff_expt_vals = False):
    """
    This module constructs the adjoint K-evolution of 
    a quantum dynamical system. It takes as parameters:
    
    ***. a reference temperature, temp_ref
    ***. the system's physical temperature, temp_rho,
    ***. a list of times for which the evolution is to
         be calculated,
    ***. the system's Hamiltonian,
    ***. a hermitian operator, whose initial ensemble-
         average is fixed via a lagrangian multiplier. 
    ***. a list of tuples of natural numbers and chosen
         operators, from which an incursive basis can be 
         constructed,
    ***. a chosen set of observables, for which their 
         expected value's time evolution is desired,
    ***. NO ESTOY SEGURO :(
    ***. (Optional): a boolean option to control whether
                     or not the projected evolution is 
                     performed.
    ***. (Optional): a boolean option for considering
                     a user-input reference state.
    ***. (Optional): a boolean option for considering a
                     reference state as a termal state,
                     this is, the exponential of the sys-
                     tem's Hamiltonian.
    ***. (Optional): a boolean option for considering an
                     initial state exactly equal to the 
                     reference state.
    ***. (Optional): a boolean option for visualizing the 
                     H-tensor's eigenvalues. 
                     Default = False.
    ***. (Optional): a boolean option for visualizing the 
                     non-hermiticity of the approximated 
                     density states. 
                     Default = False. 
    ***. (Optional): a boolean option for visualizing the
                     the observables exact and approximated
                     ensemble averages.
                     Default = True.
    ***. (Optional): a boolean option for visualizing, only,
                     the observable-wise differences between
                     the exact and the projected results.
    
    The algorithm works by constructing 
    
    1. an initial reference state, rho_ref, 
       and an endomorphism, K_ref,
    2. an incursive basis, constructed from a chosen set of 
       operators/endomorphisms by taking iterated commutators
       with the system's Hamiltonian.
    3. Said incursive basis is then orthonormalized according
       to:
           - the previously constructed reference state,
           - and either the real-valued Hilbert-Schmidt 
               inner product or the complex-valued standard
               Hilbert-Schmidt inner product.
               
    --- A check is performed to test the basis orthonormaliza-
        tion, returning an (expected) diagonal Gram matrix,
              
    4. The initial coefficient configuration is constructed 
       and its corresponding exponentiated density state. 
    5. The H-tensor is constructed from this orthonormal basis
       using the chosen version of inner product and the 
       system's Hamiltonian.
    6. From this, the time-independent markovian set of 
       coupled differential equations is solved, with the 
       H-tensor as its kernel. This yields the time-evolution of
       the coefficients and its corresponding set of density states.
    7. The observables' time evolution is calculated via ensemble 
       averages, at each desired time, using the previous density 
       states.
    8. Finally, if desired, an exact evolution is calculated using
       QuTip's master equation solver.
    
    ===> Returns a. the Gram matrix, 
                 b. the initial configurations, the reference
                     state, the initial state and the 
                     orthonormal basis.
                 c. a dictionary containing the projected 
                     time-evolution of the coefficients, the                     
                     density states and the observables' time 
                     evolution,
                 d. the exact results, obtained from QuTip's 
                    master equation solver,
    """
    start_time_proj_ev = time.time()
    beta = 1/temp_rho; beta_ref = 1/temp_ref 
    
    print("    |▼| 1. Processing reference state ===>")
    if custom_ref_state == None:
        if custom_Kref == None:
            if reference_state_choice["thermal"]:
                print("              a. ^^##^^. thermal reference state chosen")
                custom_Kref = beta_ref * Hamiltonian
                rho_ref = (-K_ref).expm()
                custom_ref_state = rho_ref/rho_ref.tr(); rho_ref == None
                assert mat_ansys.is_density_op(custom_ref_state), "reference state not a density op"
                
            if reference_state_choice["thermal+lagrange"]:
                print("              b. ^^##^^. thermal + lagrange op reference chosen")
                K_ref, rho_ref = build_reference_state(temp = temp_ref, Hamiltonian = Hamiltonian,
                                                           lagrange_op = reference_state_choice["thermal+lagrange"]["lagrange_op"], 
                                                           lagrange_mult = reference_state_choice["thermal+lagrange"]["lagrange_mult"])
        else:
            sys.exit("Error: Incompatible user-input reference state and operator")
    else:
        print("              c. ^^##^^. custom reference state chosen")
        rho_ref = custom_ref_state
        K_ref = custom_Kref
    
    basis_incursive = mat_ansys.vectorized_Hierarchical_Basis(depth_and_ops=depth_and_seed_ops,
                                                             Hamiltonian=Hamiltonian, 
                                                             rho_ref=rho_ref)
    print("    |▼| 2. Constructing Hierarchical Basis of total dimension ", len(basis_incursive)) 

    basis_orth = mat_ansys.base_orth(ops = basis_incursive, 
                           rhoref = rho_ref, 
                           sc_prod = mat_ansys.HS_inner_prod_r, 
                           visualization = False, reinforce_reality=False)   
    print("    |▼| 3. Orthonormal Hierarchical Basis of size ", len(basis_orth))
    
    ### First Test: checking if the orthonormalization has been succesful by constructing the Gram matrix
        ## and checking if said Gram matrix is the identity matrix, with a tolerance established by 
        ## default for numerical instabilities. 
    
    print("        ||♧|| Test 1. Check on the orthonormality of the HB")
    Gram_matrix = basis_orthonormality_check(basis = basis_orth, 
                                             rhoref = rho_ref, 
                                             sc_prod = mat_ansys.HS_inner_prod_r)
    
    ### constructing the initial state and H-tensor
    print("    |▼| 4. Constructing Initial Density State")
    if rho0 is None: 
        loc_coeff_list = [mat_ansys.HS_inner_prod_r(K_ref, orth_op, rhoref = rho_ref) for orth_op in basis_orth]
    else:
        loc_coeff_list = [mat_ansys.HS_inner_prod_r(K0, orth_op, rhoref = rho_ref) for orth_op in basis_orth]
    
    if rho0 is None:
        if rho_ref_equal_rho0: 
            print("            a. ^^##^^. using rho0 = rho_ref")
            phi0 = loc_coeff_list; rho0 = rho_ref    
        else: 
            print("            b. ^^##^^. constructing rho0 from the coeff. list and orth. basis")
            phi0, rho0 = build_rho0_from_basis(coeff_list = loc_coeff_list, basis = basis_orth, temp=temp_rho)
    else:
        print("            c. ^^##^^. custom Init state chosen")
        rho0 = rho0
        phi0 = loc_coeff_list
    
    ### Second Test: now, we test if the cardinality of the orthonormalized basis is equal, or not, to the dimension of the phi0 vector.
        ## The reason for this is that, at the given temperatures, the basis' depth may have saturated, with more coefficients in 
        ## the phi0-vector than operators in the orthonormalized basis. It is then recommended to run the algorithm again 
                    # 1. either changing the temp and temp_ref parameters,
                    # 2. or decreasing the hierarchical basis' depth.
                
    print("        ||♧|| Test 2. Check on the lengths of the orth. HB and the initial config phi0")
    if (len(basis_orth) != len(phi0)):
        print("            -a. ^^##^^. Error! Orthonormalization has saturated the basis. Execution stopped.")
        print("Cardinality Basis_orth=", len(basis_orth), "\n Cardinality phi0=", len(phi0))
        sys.exit()
    else:
        print("            -b. ^^##^^. Check passed: phi0 and orth. HB have the same cardinalities.")
    
    print("    |▼| 5. Constructing H-tensor from orth. HB")
    Hijtensor = H_ij_matrix(Hamiltonian = Hamiltonian,
                            basis = basis_orth, 
                            rhoref = rho_ref, 
                            sc_prod = mat_ansys.HS_inner_prod_r)
    
    ### constructing the coefficient arrays and the physical states
    res_proj_ev_rhot_list, phit_list, tracerho_at_timet = semigroup_phit_and_rhot_sol(phi0 = phi0, 
                                                                                       rho0 = rho0, 
                                                                                       Htensor = Hijtensor, 
                                                                                       ts = timespan, 
                                                                                       basis = basis_orth)
     ### test 3
    herm_rhot_list = semigroup_rhos_test(rho_list = res_proj_ev_rhot_list, 
                                         visualization_nonherm = visualization_nonherm, 
                                         ts = timespan)
   
    res_proj_ev_obs_list = [np.array([qutip.expect(obs, rhot) for rhot in herm_rhot_list]) for obs in observables]
    proj_ev_runtime = time.time() - start_time_proj_ev
    
    print("    |▼| 6. ProjEv Dynamics Succesfully Concluded.")
    
    ### Exact solution 
    
    if compute_exact_dynamics: 
        print("    |▼| 7a. Computing Exact Dynamics with QuTip package.")
        start_time_exact = time.time()
        res_exact = mod_mesolve(Hamiltonian, rho0=rho0, tlist=timespan, c_ops=None, e_ops=observables)
        assert rho0 == res_exact.states[0], "Error: Exact initial state != Proj-ev initial state"
        exact_ev_runtime = time.time() - start_time_exact
    else:
        print("    |▼| 7b. Exact Dynamics not to be computed. Skipped.")
        res_exact = None
    
    init_configs = {}; 
    init_configs["Gram matrix"] = Gram_matrix; init_configs["rho_ref"] = rho_ref
    init_configs["rho0"] = rho0; init_configs["basis_orth"] = basis_orth
    
    init_configs["proj_ev_runtime"] = proj_ev_runtime
    if compute_exact_dynamics: 
        init_configs["exact_ev_runtime"] = exact_ev_runtime
    
    res_proj_ev = {}
    res_proj_ev["H_tensor"] = Hijtensor
    res_proj_ev["Coeff_ev"] = phit_list
    res_proj_ev["State_ev"] = herm_rhot_list; 
    res_proj_ev["Avgs"] = res_proj_ev_obs_list
    res_proj_ev["Traces"] = tracerho_at_timet
    
    if visualize_expt_vals and compute_exact_dynamics:
        print("    |▼| 7a. Processing ProjEv v. Exact Plots.")
        evs_plot.plot_exact_v_proj_ev_avgs(obs = observables, labels = label_ops, timespan = timespan, 
                                  Result_proj_ev = res_proj_ev["Avgs"], 
                                  Result_exact = res_exact, 
                                  visualize_diff_expt_vals = visualize_diff_expt_vals
                                  )
        label_metric = ["Bures Exact v. Proj ev", "S(exact || proj_ev)", "S(proj_ev || exact)"]
        metric_local = exact_v_proj_ev_matrix_metrics(ts, res_proj_ev["State_ev"], res_exact)
        evs_plot.evs_plot.plot_exact_v_proj_ev_metrics(timespan, 
                                     res_proj_ev["State_ev"], 
                                     res_exact, 
                                     label_metric
                                    )
    else:
        print("    |▼| 7b. No Plots to process.")
        
    evolutions_data = {}
    evolutions_data["initial_configs_evs"] = init_configs
    evolutions_data["proj_ev"] = res_proj_ev
    evolutions_data["exact_ev"] = res_exact
    
    coeff_list = None; loc_coeff_list = None
    Gram_matrix = None; rho_ref = None; rho0 = None; basis_orth = None
    Hijtensor = None; phit_list = None; herm_rhot_list = None; res_proj_ev_obs_list = None
    
    print("    |▼| 8. Data Stored. Evolutions concluded. \n")
    return evolutions_data 