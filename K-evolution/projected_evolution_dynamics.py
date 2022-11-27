# In [1]:

import qutip, sys, pickle
import numpy as np
import scipy.optimize as opt 
import matplotlib.pyplot as plt
import time as time
import scipy.linalg as linalg

import matrix_analysis_lib as mat_ansys
import spin_representations as su2
import evs_visualization_tools as evs_plot

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

def H_ij_matrix(Hamiltonian, basis, rho0, sc_prod):
    coeffs_matrix = np.array([[sc_prod(op1, -1j * mat_ansys.commutator(Hamiltonian, op2), rho0) for op2 in basis] for op1 in basis])
    return coeffs_matrix

def basis_orthonormality_check(basis, rho0, sc_prod, visualization_Gram_m = False): 
    dim = len(basis)
    hermitian_basis = [mat_ansys.non_hermitianess_measure(op1) <= 1e-10 for op1 in basis]
    assert np.all(hermitian_basis), ("Not all the operators are "
                                     f"hermitian:\n {hermitian_basis}")
    
    gram_matrix = [[sc_prod(op2, op1, rho0) for op2 in basis] for op1 in basis]
    normalized = [abs(gram_matrix[i][i]-1.) <= 1e-10  for i in range(dim)]
    assert all(normalized), ("Some operators in the basis are not normalized:\n"
                             f"{normalized}")

    assert (linalg.norm((np.identity(dim) - gram_matrix) <= 1e-10)), "Not all operators are pair-wise orthogonal"
    print("The basis is orthonormal and hermitian")

    null_averages = [np.real((rho0 * op1).tr()) <= 1e-6 for op1 in basis]
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
        if not (np.linalg.norm( (K-K.dag()).full()) < 1e-5):
            print("Non hermitician part norm:", np.linalg.norm( (K-K.dag()).full())  )
            assert K.isherm, "K is not Hermitician "
        rhot= K.expm()
        rho_at_timet.append(rhot/rhot.tr())
    return rho_at_timet, Phi_vector_solution    

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

def d_depth_proj_ev(temp_ref, temp_rho, timespan, Hamiltonian, lagrange_op,
                    depth_and_seed_ops, observables, label_ops, coeff_list = None, 
                    custom_ref_state = None, 
                    rho_ref_thermal_state = False, 
                    rho_ref_equal_rho0 = False, 
                    visualize_H_evs = False, 
                    visualization_nonherm = False, 
                    visualize_expt_vals = True, 
                    visualize_diff_expt_vals = False):
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
    ### Projected solutions
    ### building reference states and testing it
    
    start_time_proj_ev = time.time()
    print("1. Processing reference state ===>")
    if custom_ref_state is None:
        if rho_ref_thermal_state:
            print("    ^^##. thermal reference state chosen")
            beta_ref = 1/temp_ref
            K_ref = beta_ref * Hamiltonian 
            rho_ref = (K_ref).expm()
            rho_ref = rho_ref/rho_ref.tr()
            assert mat_ansys.is_density_op(rho_ref), "reference state not a density op"
        
        else:
            print("    ^^##. thermal reference state with Lagrange multiplier chosen")
            K_ref, rho_ref = build_reference_state(temp = temp_ref, 
                                                   Hamiltonian = Hamiltonian,
                                                   lagrange_op = lagrange_op, 
                                                   lagrange_mult = .5)
    else:
        print("    ^^##. custom reference state chosen")
        rho_ref = custom_ref_state
    
    basis_incursive = mat_ansys.vectorized_recursive_basis(depth_and_ops=depth_and_seed_ops,                                             
                                                 Hamiltonian=Hamiltonian, 
                                                 rho0=rho_ref)
    
    basis_orth = mat_ansys.base_orth(ops = basis_incursive, 
                           rho0 = rho_ref, 
                           sc_prod = mat_ansys.HS_inner_prod_r, 
                           visualization = False, reinforce_reality=False)   
    print("2. using a base of size ", len(basis_orth))
    print("3. rho_ref: ", rho_ref)
    
        ### test 
    Gram_matrix = basis_orthonormality_check(basis = basis_orth, 
                                             rho0 = rho_ref, 
                                             sc_prod = mat_ansys.HS_inner_prod_r)
    
        ### constructing the initial state and H-tensor
    
    loc_coeff_list = coeff_list
    
    if rho_ref_equal_rho0: 
        print("3. using rho0 = rho_ref")
        phi0 = loc_coeff_list; rho0 = rho_ref    
    else: 
        print("3. constructing rho0 from the coeff. list and orth. basis")
        phi0, rho0 = build_rho0_from_basis(coeff_list = loc_coeff_list, basis = basis_orth, temp=temp_rho)
        
    Hijtensor = H_ij_matrix(Hamiltonian = Hamiltonian,
                            basis = basis_orth, 
                            rho0 = rho_ref, 
                            sc_prod = mat_ansys.HS_inner_prod_r)
    
    ### constructing the coefficient arrays and the physical states
    res_proj_ev_rhot_list, phit_list = semigroup_phit_and_rhot_sol(phi0 = phi0, 
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
    
    ### Exact solution 
    start_time_exact = time.time()
    res_exact = mod_mesolve(Hamiltonian, rho0=rho0, tlist=timespan, c_ops=None, e_ops=observables)
    
    assert rho0 == res_exact.states[0], "Error: Exact initial state != Proj-ev initial state"
    exact_ev_runtime = time.time() - start_time_exact
    
    initial_configs = {}; 
    initial_configs["Gram matrix"] = Gram_matrix; initial_configs["rho_ref"] = rho_ref
    initial_configs["rho0"] = rho0; initial_configs["basis_orth"] = basis_orth
    
    dict_res_proj_ev = {}
    dict_res_proj_ev["H_tensor"] = Hijtensor
    dict_res_proj_ev["Coeff_ev"] = phit_list
    dict_res_proj_ev["State_ev"] = herm_rhot_list; 
    dict_res_proj_ev["Avgs"] = res_proj_ev_obs_list
    
    if visualize_expt_vals:
        evs_plot.plot_exact_v_proj_ev_avgs(obs = observables, labels = label_ops, timespan = timespan, 
                                  Result_proj_ev = dict_res_proj_ev["Avgs"], 
                                  Result_exact = res_exact, 
                                  visualize_diff_expt_vals = visualize_diff_expt_vals
                                  )
        label_metric = ["Bures Exact v. Proj ev", "S(exact || proj_ev)", "S(proj_ev || exact)"]
        metric_local = exact_v_proj_ev_matrix_metrics(ts, dict_res_proj_ev["State_ev"], res_exact)
        evs_plot.evs_plot.plot_exact_v_proj_ev_metrics(timespan, 
                                     dict_res_proj_ev["State_ev"], 
                                     res_exact, 
                                     label_metric
                                    )
    
    evs_data = {}
    evs_data["proj_ev_runtime"] = proj_ev_runtime
    evs_data["exact_ev_runtime"] = exact_ev_runtime
    
    coeff_list = None; loc_coeff_list = None
    Gram_matrix = None; rho_ref = None; rho0 = None; basis_orth = None
    Hijtensor = None; phit_list = None; herm_rhot_list = None; res_proj_ev_obs_list = None
    
    print("4. Evolutions concluded.")
    
    return initial_configs, evs_data, dict_res_proj_ev, res_exact

def N_fixed_multiple_temps_proj_evs(depth_and_seed_ops, observables, label_ops, 
                                    ref_operator,
                                    Hamiltonian,
                                    temp_rho, 
                                    init_coeff_list,
                                    timespan, 
                                    range_temps):
    
    """
    This module performs multiple projected and exact
    evolutions for different reference temperatures, 
    at fixed spin chain-length and fixed Lie algebra
    dimension. This module takes as input:
    
        *♥*♥* 1. depth_and_seed_ops: 
                    a chosen set of operators/endomorphisms.
                    In our context, this set will be constructed
                    by taking iterated commutators with the
                    system's Hamiltonian, yielding a Lie algebra. 
                    
        *♥*♥* 2. observables: 
                    a chosen set of observables, for which their 
                    expected value's time evolution is desired.
                    
        *♥*♥* 3. label_ops: 
                    a list with strings values, containing the 
                    observables' physical name. These string 
                    literals are used for plotting the results.
                    
        *♥*♥* 4. ref_operator: 
                    a hermitian operator, from which a gaussian 
                    reference state may be constructed, using some 
                    user-input reference temperatures (see point 8). 
                    This state is constructed as:
                         
                         rho_ref = (-1/temp_ref * ref_operator).expm().
                    
        *♥*♥* 5. Hamiltonian:
                    the system's Hamiltonian. 
                    
        *♥*♥* 6. temp_rho: 
                    a fixed temperature for the system's initial 
                    state. 
        
        *♥*♥* 7. init_coeff_list:
                    Given an orthonormal basis of operators, this 
                    list of real numbers provides the initial 
                    state's decomposition in terms of said basis. 
                 
        *♥*♥* 8. timespan:
                    an array of times for which the time evolution
                    of the observables is to be calculated.
                     
        *♥*♥* 9. range_temps: 
                     a list of reference temperatures for which the 
                     different reference operators are constructed, as
                     thermal-like states, ie.
                     
                         rho_ref = (-1/temp_ref * ref_operator).expm()
                        
                    For each one of these refrence operators, both the
                    exact and projected evolutions are computed and returned. 
                    
        ====> Returns: multiple_evolutions: a dictionary storing all the 
                                            evolutions' data. 
                                            
             
        How the algorithm works: For a fixed-length spin chain and fixed
                                 set of operators, "depth_and_seed_ops" 
                                -constructed from iterated commutators-, 
        
            (i)*. a user-input set of reference temperatures, *range_temps*,
                  is processed. For each one of these temperatures 
                  
                  .a: a thermal-like state, 'custom_rho_ref', is constructed 
                          using the user-input parameter "reference_operator".
                          A check is performed to confirm whether or not it is
                          a valid density operator. 
                          
                  .b: the projected and exact evolutions are computed, by calling
                      the d_depth_proj_ev routine, which takes as input (amongst others)
                      the previous reference state, the system's Hamiltonian, 
                      the list of observables. 
                  
                  .c: these results are then stored in four dictionaries 
                      (see the graph for a pictoral
                          representation of the data's structure). 
                      
            (ii)*. These steps are made for every desired reference temperature. 
                   When all of these are processed and their evolutions concluded,
                   a single dictionary, with the previous four nested dictionaries, 
                   is returned. 
       
       Warnings: (A). This module assumes all user input operator-valued parameters 
                      are hermitian. 
                 (B). This module assumes all operators have compatible dimensions 
                      as well. 
                 (C). All reference operators undergo a test for determining if these
                      are density operators of not. Should one of these tests fail,
                      the module will crash and all date will be lost.                
                 (D). Numerical stability has been checked upto temperatures of e-2
                      order. Lower temperatures may have numerical instabilities 
                      crashing the calculations.          
    """
    
    
    labels = ["Temp_" + str(i) for i in range(len(range_temps))]
    multiple_evolutions = {}
    multiple_init_configs = {}; multiple_evs_data = {}; multiple_dict_res_proj_ev = {}; multiple_res_exact = {}
        
    for Temp_Ref in range_temps:
        print("Processing step: ", range_temps.index(Temp_Ref), "and temperature ", Temp_Ref)
        
        loc_coeff_list = init_coeff_list
        
        beta_ref = (1/Temp_Ref)
        K_ref = - beta_ref * .5 * ref_operator
        rho_ref = (K_ref).expm()
        custom_rho_ref = rho_ref/rho_ref.tr()
        
        assert mat_ansys.is_density_op(custom_rho_ref), "Error: rho_ref is not a density operator"
        
        init_configs_MFT_state, evs_data, dict_res_proj_ev, res_exact = d_depth_proj_ev(
                temp_ref = Temp_Ref, temp_rho = temp_rho, 
                timespan = timespan, 
                Hamiltonian = Hamiltonian, lagrange_op = None,
                depth_and_seed_ops = depth_and_seed_ops, observables = observables, 
                label_ops = label_ops, coeff_list = loc_coeff_list, 
                custom_ref_state = custom_rho_ref, 
                rho_ref_thermal_state = False,
                rho_ref_equal_rho0 = False, visualize_H_evs = False, 
                visualization_nonherm = False, visualize_expt_vals = False, visualize_diff_expt_vals = False
                )
            
        multiple_init_configs["init_configs_T" + str(range_temps.index(Temp_Ref))] = init_configs_MFT_state
        multiple_evs_data["evs_data_T" + str(range_temps.index(Temp_Ref))] = evs_data
        multiple_dict_res_proj_ev["dict_res_proj_ev_T" + str(range_temps.index(Temp_Ref))] = dict_res_proj_ev
        multiple_res_exact["res_exact_T" + str(range_temps.index(Temp_Ref))] = res_exact
                        
    multiple_evolutions["init_configs_all"] = multiple_init_configs
    multiple_evolutions["evs_data_all"] = multiple_evs_data
    multiple_evolutions["dict_res_proj_ev_all"] = multiple_dict_res_proj_ev
    multiple_evolutions["res_exact_all"] = multiple_res_exact
            
    return multiple_evolutions

def temp_fixed_multiple_dims_proj_evs(chain_type, Hamiltonian_paras,
                                      derived_series_op_order, 
                                      temp_ref, temp_rho, 
                                      init_coeff_list,
                                      timespan, range_dims, ref_operator_type = "Mean_field_state"):
    
    """
    This module performs multiple projected and exact
    evolutions for different spin chain lengths temperatures, 
    at fixed (reference and initial) temperatures and fixed 
    Lie algebra. This module takes as input:
    
        *♥*♥* 1. chain_type: 
                    the desired type of Heisenberg spin chain is chosen. 
                    In this version, the allowed chain types are:
                    "XX", "XYZ", "XXZ", "XXX", "Anderson" spin chains.
        
        *♥*♥* 2. Hamiltonian_paras: 
                    the parameters and weights of the different terms
                    in the Heisenberg Hamiltonian. For further info 
                    and detail, check the Heisenberg_Hamiltonian module.
                    
        *♥*♥* 3. derived_series_op_order:
                    a natural number indicating how many iterated 
                    commutators will form part of the Lie basis.
        
        *♥*♥* 4. temp_ref: 
                    a fixed temperature for the system's reference state.
        
        *♥*♥* 5. temp_rho: 
                    a fixed temperature for the system's initial  state.
                    
        *♥*♥* 6. init_coeff_list:
                    Given an orthonormal basis of operators, this 
                    list of real numbers provides the initial 
                    state's decomposition in terms of said basis. 
        
        *♥*♥* 7. timespan:
                    an array of times for which the time evolution
                    of the observables is to be calculated.
                    
        *♥*♥* 8. range_dims: 
                    a list of spin chain lengths. For each of these lenghts, 
                    both the exact and projected evolutions are computed 
                    and returned. 
                    
        *♥*♥* 9. ref_operator_type:
                    a string-literal option to control which type of 
                    thermal-like reference operator is to constructed.
                    Its default value is Mean_field_state" and the following
                    reference state is constructed: 
                    
                         rho_ref = (- beta_ref * .5 * spin_ops_list[1][0]).expm()
        
        ====> Returns: multiple_evolutions: a dictionary storing all the 
                                            evolutions' data. 
                                            
             
        How the algorithm works: For a fixed type of Heisenberg spin chain and
                                 fixed set of operators, "depth_and_seed_ops" 
                                -constructed from iterated commutators-, 
        
            (i)*. a user-input set of spin chain lengths, "range_dims",
                  is processed. For each one of these lengths 
                  
                  .a: the systems Hamiltonian is constructed,
                  
                  .b: a set of classical operators is constructed as well, 
                      along with their labels. 
                      
                  .c: the projected and exact evolutions are computed, by calling
                      the d_depth_proj_ev routine, which takes as input (amongst others)
                      the previous reference state, the system's Hamiltonian, 
                      the list of observables. 
                  
                  .d: these results are then stored in four dictionaries 
                      (see the graph for a pictoral
                          representation of the data's structure). 
                      
            (ii)*. These steps are made for every desired chain length. 
                   When all of these are processed and their evolutions concluded,
                   a single dictionary, with the previous four nested dictionaries, 
                   is returned. 
       
       Warnings: (A). This module assumes all user-input operator-valued parameters 
                      are hermitian. 
                 (B). This module assumes all operators have compatible dimensions 
                      as well. s
                 (C). All reference operators undergo a test for determining if these
                      are density operators of not. Should one of these tests fail,
                      the routine will crash and all previously gathered data will be lost.                
                 (D). Numerical stability has been checked upto temperatures of e-2
                      order and chain size of 10 spins. 
                      Lower temperatures and higher spin chain lengths may have numerical instabilities 
                      crashing the calculations.          
    """

    labels = ["dim_" + str(i) for i in range(len(range_dims))]
    multiple_evolutions = {}
    multiple_init_configs = {}; multiple_evs_data = {}; multiple_dict_res_proj_ev = {}; multiple_res_exact = {}

    for length in range_dims:
        #try: 
            print("Processing step: ", range_dims.index(length), "and spin-chain of length  ", length)
            spin_ops_list = su2.one_body_spin_ops(length)
            identity_op = spin_ops_list[0][0]
    
            Hamiltonian = su2.Heisenberg_Hamiltonian(op_list = spin_ops_list, chain_type = chain_type,
                                                     size = length, Hamiltonian_paras = Hamiltonian_paras,
                                                     closed_bcs = True, visualization = False)
        
            cl_ops, label_ops = su2.classical_ops(Hamiltonian, length, spin_ops_list, False)
            magnetization = sum(spin_ops_list[3][a] for a in range(len(spin_ops_list[3])))
            neel_operator = sum((-1)**a * spin_ops_list[3][a] for a in range(len(spin_ops_list[3])))
            cl_ops["magnetization"] = magnetization; label_ops.append("magnetization")
            cl_ops["neel_op"] = neel_operator; label_ops.append("neel_op")
            assert mat_ansys.basis_hermitian_check(cl_ops), "Not all operators are Hermitian"
            
            beta_ref = (1/temp_ref)
            if ref_operator_type == "Mean_field_state":
                
                depth_and_seed_ops = [(1, cl_ops["identity_op"]), 
                      (1, Hamiltonian), 
                      (derived_series_op_order, spin_ops_list[1][0]),
                      ]
                
                K_ref = - beta_ref * .5 * spin_ops_list[1][0]
                
            rho_ref = (K_ref).expm()
            custom_rho_ref = rho_ref/rho_ref.tr(); rho_ref = None
            
            assert mat_ansys.is_density_op(custom_rho_ref), "Reference State not density op"
        
            init_configs_MFT_state, evs_data, dict_res_proj_ev, res_exact = d_depth_proj_ev(
                    temp_ref = temp_ref, temp_rho = temp_rho, 
                    timespan = timespan, 
                    Hamiltonian = Hamiltonian, lagrange_op = None,
                    depth_and_seed_ops = depth_and_seed_ops, observables = list(cl_ops.values()),
                    label_ops = label_ops, coeff_list = init_coeff_list, 
                    custom_ref_state = custom_rho_ref, 
                    rho_ref_thermal_state = False, rho_ref_equal_rho0 = False, visualize_H_evs = False, 
                    visualization_nonherm = False, visualize_expt_vals = False, visualize_diff_expt_vals = False
                    )
       
            multiple_init_configs["init_configs_N" + str(range_dims.index(length))] = init_configs_MFT_state
            multiple_evs_data["evs_data_N" + str(range_dims.index(length))] = evs_data
            multiple_dict_res_proj_ev["dict_res_proj_ev_N" + str(range_dims.index(length))] = dict_res_proj_ev
            multiple_res_exact["res_exact_N" + str(range_dims.index(length))] = res_exact
            
        #except Exception as ex:
        #        print(ex)
        #        break
        
    multiple_evolutions["init_configs_all"] = multiple_init_configs
    multiple_evolutions["evs_data_all"] = multiple_evs_data
    multiple_evolutions["dict_res_proj_ev_all"] = multiple_dict_res_proj_ev
    multiple_evolutions["res_exact_all"] = multiple_res_exact
    
    return multiple_evolutions

def increase_depth_multiple_proj_evs(Hamiltonian, rho_ref, range_derived_series_orders, 
                                     temp_ref, temp_rho,
                                     generating_operator,
                                     init_coeff_list,
                                     timespan, label_ops,
                                     observables):
    
    """
    This module performs multiple projected and exact
    evolutions for different reference temperatures, 
    at fixed spin chain-length and fixed Lie algebra
    dimension. This module takes as input:
    
        *♥*♥* 1. depth_and_seed_ops: 
                    a chosen set of operators/endomorphisms.
                    In our context, this set will be constructed
                    by taking iterated commutators with the
                    system's Hamiltonian, yielding a Lie algebra. 
                    
        *♥*♥* 2. observables: 
                    a chosen set of observables, for which their 
                    expected value's time evolution is desired.
                    
        *♥*♥* 3. label_ops: 
                    a list with strings values, containing the 
                    observables' physical name. These string 
                    literals are used for plotting the results.
                    
        *♥*♥* 4. ref_operator: 
                    a hermitian operator, from which a gaussian 
                    reference state may be constructed, using some 
                    user-input reference temperatures (see point 8). 
                    This state is constructed as:
                         
                         rho_ref = (-1/temp_ref * ref_operator).expm().
                    
        *♥*♥* 5. Hamiltonian:
                    the system's Hamiltonian. 
                    
        *♥*♥* 6. temp_rho: 
                    a fixed temperature for the system's initial 
                    state. 
        
        *♥*♥* 7. init_coeff_list:
                    Given an orthonormal basis of operators, this 
                    list of real numbers provides the initial 
                    state's decomposition in terms of said basis. 
                 
        *♥*♥* 8. timespan:
                    an array of times for which the time evolution
                    of the observables is to be calculated.
                     
        *♥*♥* 9. range_temps: 
                     a list of reference temperatures for which the 
                     different reference operators are constructed, as
                     thermal-like states, ie.
                     
                         rho_ref = (-1/temp_ref * ref_operator).expm()
                        
                    For each one of these refrence operators, both the
                    exact and projected evolutions are computed and returned. 
                    
        ====> Returns: multiple_evolutions: a dictionary storing all the 
                                            evolutions' data. 
                                            
             
        How the algorithm works: For a fixed-length spin chain and fixed
                                 set of operators, "depth_and_seed_ops" 
                                -constructed from iterated commutators-, 
        
            (i)*. a user-input set of reference temperatures, "range_temps",
                  is processed. For each one of these temperatures 
                  
                  .a: a thermal-like state, 'custom_rho_ref', is constructed 
                          using the user-input parameter "reference_operator".
                          A check is performed to confirm whether or not it is
                          a valid density operator. 
                          
                  .b: the projected and exact evolutions are computed, by calling
                      the d_depth_proj_ev routine, which takes as input (amongst others)
                      the previous reference state, the system's Hamiltonian, 
                      the list of observables. 
                  
                  .c: these results are then stored in four dictionaries 
                      (see the graph for a pictoral
                          representation of the data's structure). 
                      
            (ii)*. These steps are made for every desired reference temperature. 
                   When all of these are processed and their evolutions concluded,
                   a single dictionary, with the previous four nested dictionaries, 
                   is returned. 
       
       Warnings: (A). This module assumes all user input operator-valued parameters 
                      are hermitian. 
                 (B). This module assumes all operators have compatible dimensions 
                      as well. 
                 (C). All reference operators undergo a test for determining if these
                      are density operators of not. Should one of these tests fail,
                      the module will crash and all date will be lost.                
                 (D). Numerical stability has been checked upto temperatures of e-2
                      order. Lower temperatures may have numerical instabilities 
                      crashing the calculations.          
    """
    
    multiple_evolutions = {}
    multiple_init_configs = {}; multiple_evs_data = {}; multiple_dict_res_proj_ev = {}; multiple_res_exact = {}
    
    for deg_solva in range_derived_series_orders:
        print("Processing step: ", range_derived_series_orders.index(deg_solva)+1, " and Lie subalgebra of dim ", deg_solva)
        
        id_op = qutip.tensor([qutip.qeye(2) for k in (Hamiltonian.dims[0])])
        depth_and_seed_ops = [(1, id_op), 
                              (1, Hamiltonian), 
                              (deg_solva+1, generating_operator)]
        
        init_configs_MFT_state, evs_data, dict_res_proj_ev, res_exact = d_depth_proj_ev(
                    temp_ref = temp_ref, temp_rho = temp_rho, 
                    timespan = timespan, 
                    Hamiltonian = Hamiltonian, lagrange_op = None,
                    depth_and_seed_ops = depth_and_seed_ops, observables = observables,
                    label_ops = label_ops, coeff_list = init_coeff_list[range_derived_series_orders.index(deg_solva)], 
                    custom_ref_state = rho_ref, 
                    rho_ref_thermal_state = False, rho_ref_equal_rho0 = False, visualize_H_evs = False, 
                    visualization_nonherm = False, visualize_expt_vals = False, visualize_diff_expt_vals = False
                    )
       
        multiple_init_configs["init_configs_Liedim" + str(range_derived_series_orders.index(deg_solva)+1)] = init_configs_MFT_state
        multiple_evs_data["evs_data_Liedim" + str(range_derived_series_orders.index(deg_solva)+1)] = evs_data
        multiple_dict_res_proj_ev["dict_res_proj_ev_Liedim" + str(range_derived_series_orders.index(deg_solva)+1)] = dict_res_proj_ev
        multiple_res_exact["res_exact_Liedim" + str(range_derived_series_orders.index(deg_solva)+1)] = res_exact
                    
    multiple_evolutions["init_configs_all"] = multiple_init_configs
    multiple_evolutions["evs_data_all"] = multiple_evs_data
    multiple_evolutions["dict_res_proj_ev_all"] = multiple_dict_res_proj_ev
    multiple_evolutions["res_exact_all"] = multiple_res_exact
    
    return multiple_evolutions
