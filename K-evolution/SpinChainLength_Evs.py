def temp_fixed_multiple_dims_proj_evs(chain_type, Hamiltonian_paras,
                                      derived_series_op_order, 
                                      temp_ref, temp_rho, 
                                      init_coeff_list,
                                      timespan, range_dims, ref_operator_type = "Mean_field_state"):
    
    """
    This module performs multiple projected and exact
    evolutions for different spin chain lengths, 
    at fixed (reference and initial) temperatures and fixed 
    hierarchical basis. This module takes as input:
    
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
