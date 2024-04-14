#In [0]

import qutip
import numpy as np
import scipy.linalg as linalg
import a_quantum_geometries as gij 

from datetime import datetime
from typing import Callable, List, Optional, Tuple, Dict
from qutip import Qobj

# In [1]:

def get_function_from_list(Hme: List[Qobj]): 
    for d in Hme:
        sum

def post_ev_projections(exact_Ks: List[Qobj], exact_states:List[Qobj], 
                        K0:Qobj, basis:List[Qobj], tlist:np.array, args: Dict, verbose=True, ):
    
    print("**** Starting Simulation", datetime.now())
    results={};
    results['covar_Kt']=[]; results['covar_sigmat']=[]
    
    covar_sp=[gij.fetch_covar_scalar_product(rho_at_timet) for rho_at_timet in exact_states]
    timespan_prime=list(tlist)
    for t in tlist:
        ti = timespan_prime.index(t)
        d=int(.20*len(tlist))
        if ti%d==0:
            print("   -----Processing step",ti,"at time",datetime.now)
        orth_basis=gij.orthogonalize_basis(basis=basis,sp=covar_sp[ti],
                                           idop=qutip.tensor([qutip.qeye(2) for dim in exact_states[0].dims[0]]))
        if verbose and len(orth_basis) != len(basis)+1:
            if ti%d==0:
                print('→→→ Some operators were discarded. Orth. Basis length:', len(orth_basis))
        phis_covar = gij.project_op(exact_Ks[ti], orth_basis, covar_sp[ti])
        Kp_covar=sum(phi*op for phi,op in zip(phis_covar, orth_basis))
        sigma_covar=gij.safe_expm_and_normalize(Kp_covar)
        results['covar_Kt'].append(Kp_covar)
        results['covar_sigmat'].append(sigma_covar)
    return results

# In [2]:

def multiple_projections(exact_Ks: List[Qobj], exact_states: List[Qobj], 
                         generator, K0: Qobj, basis0: List[Qobj], depths: List[int], 
                         timespan: np.array, args: Dict, td_generator: bool=True, verbose=True, magnus=True):
    """
    Perform multiple projections with varying depths of hierarchical basis.

    Args:
        exact_Ks (List[Qobj]): List of exact K operators.
        exact_states (List[Qobj]): List of exact states.
        generator (Qobj): Generator.
        K0 (Qobj): Initial K operator.
        basis0 (List[Qobj]): Initial basis.
        depths (List[int]): List of depths to use.
        timespan (np.array): Array of times.

    Returns:
        Tuple: Tuple containing lists of projected Ks and density states.
    """
    res_sigmas = []
    res_Ks = []

    for d in depths:
        if td_generator:
            period=args.get('period')
            local_avg_timespan=np.linspace(0,period, int(period)*10)
            avg_generator=1/period*sum(generator(t=ti, args=args) for ti in local_avg_timespan)*(local_avg_timespan[1]-local_avg_timespan[0])
            if magnus:
                avg_generator+=gij.magnus_1t(generator=generator, args=args)
        else:
            avg_generator=generator
             
        try:
            hb_basis_d = gij.build_HierarchicalBasis(generator=avg_generator, seed_operator=basis0[-1],
                                                     depth=d)
        except Exception as e:
            print('Error with the Hierarchical Basis:', e)
            hb_basis_d = None

        if hb_basis_d is not None:
            try: 
                res_d = post_ev_projections(exact_Ks=exact_Ks,
                                            exact_states=exact_states, 
                                            K0=K0, 
                                            basis=hb_basis_d,
                                            tlist=timespan,
                                            args=args)
            except Exception as e:
                print('Error with the Dynamics:', e)
                res_d = None
        else:
            print("Error: Basis not correctly constructed")
            res_d = None
           
        res_sigmas.append(res_d['covar_sigmat']) if res_d else res_sigmas.append(None)
        res_Ks.append(res_d['covar_Kt']) if res_d else res_Ks.append(None)

    return res_sigmas, res_Ks
