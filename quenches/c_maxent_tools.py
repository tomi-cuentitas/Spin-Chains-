#In [0]

import qutip
import numpy as np
import scipy.linalg as linalg
import a_quantum_geometries as gij 

from datetime import datetime
from typing import Callable, List, Optional, Tuple
from qutip import Qobj

# In [1]:

def post_ev_projections(exact_Ks:List[Qobj], exact_states:List[Qobj], K0:Qobj, basis:List[Qobj], tlist:np.array, args:dict, verbose=True):
    print("**** Starting Simulation", datetime.now())
    results={};
    results['covar_Kt']=[]; results['covar_sigmat']=[]
    
    covar_sp=[gij.fetch_covar_scalar_product(rho_at_timet) for rho_at_timet in exact_states]
    timespan_prime=list(tlist)
    for t in tlist:
        ti = timespan_prime.index(t)
        d=int(.1*len(tlist))
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

def multiple_projections(exact_Ks, exact_states, generator, K0, basis0, depths, timespan, args):
                         
    res_sigmas=[]; res_Ks=[]
    for d in depths:
        try:
            hb_basis_d=gij.build_HierarchicalBasis(generator=generator,
                                               seed_operator=basis0[-1],
                                               depth=d)
        except Error:
            print('Error with the Hierarchical Basis')
            hb_basis_d=None

        if type(hb_basis_d) != None:
            try: 
                res_d=post_ev_projections(exact_Ks=exact_Ks,
                                                       exact_states=exact_states, 
                                                       K0=K0, 
                                                       basis=hb_basis_d,
                                                       tlist=timespan,
                                                       args=args)
            except Error:
                    print('Error with the Dynamics')
                    res_d=None
        else:
            res_d=None
           
        res_sigmas.append(res_d['covar_sigmat']); res_Ks.append(res_d['covar_Kt'])
        res_d=None
        
    return res_sigmas, res_Ks
    
