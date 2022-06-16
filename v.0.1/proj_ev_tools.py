#!/usr/bin/env python
# coding: utf-8

# In[1]:

import qutip
import numpy
import scipy.optimize as opt 
import pickle


# In[2]:


def prod_basis(b1, b2):
  return [qutip.tensor(b,s) for b in b1 for s in b2]

def scalar_prod(op1, op2, rho0 = None):
  if op1.dims[0][0]!=op2.dims[0][0]:
    return "Incompatible Qobj dimensions"
  if rho0 is None:
    rho0 = qutip.qeye(op1.dims[0])/op1.dims[0][0]
  result = .5*(rho0*(op1*op2.dag()+op2.dag()*op1)).tr()
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


# In[3]:


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
    res = opt.minimize(test,np.zeros(len(basis)),args=(rho,basis))
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
      print ("orth error")
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


# In[4]:


id2 = qutip.qeye(2)
sx = .5*qutip.sigmax()
sy = .5*qutip.sigmay()
sz = .5*qutip.sigmaz()
globalid = qutip.tensor([qutip.qeye(2) for k in range(N)])


sx_list = []
sy_list = []
sz_list = []