import time, os, glob
import scipy
import numpy as np
import matplotlib.pyplot as plt
import emcee

import cyclic_hd


def forward_calculation(G0,gamma_ref,gamma_ref_h, shear_strain):
    clay = cyclic_hd.Cyclic_HD(G0,gamma_ref,gamma_ref_h,Yoshida=True)
    shear_stress = clay.cyclic_shear(shear_strain)
    return shear_stress

def residual_function(param,strain_list,stress_list):
    G0,gamma_ref,gamma_ref_h = param
    res = 0.0
    for i,strain in enumerate(strain_list):
        var = np.var(stress_list[i])
        stress_calc = forward_calculation(G0,gamma_ref,gamma_ref_h,strain)
        res += np.sum((stress_list[i] - stress_calc)**2)/var

    return res

def ln_residual_function(param,strain_list,stress_list):
    if param[0] <= 0.0:
        return -np.inf
    if param[1] <= 0.0:
        return -np.inf
    if param[2] <= 0.0:
        return -np.inf
    return -residual_function(param,strain_list,stress_list)

#### Main function ####
# --- Read data list --- #
data_files = glob.glob("result/*.dat")
strain_list, stress_list = [],[]
for data_file in data_files:
    strain,stress = np.loadtxt(data_file,usecols=(0,2),unpack=True)
    strain_list += [strain]
    stress_list += [stress]

# --- Set initial value --- #
ndim = 3
nwalkers = 50
nstep = 500

G0 = 1.e6
gamma_ref = 1.e-3
gamma_ref_h = 1.e-3

x0 = np.random.lognormal(np.log(G0),1,size=nwalkers)
x1 = np.random.lognormal(np.log(gamma_ref),1,size=nwalkers)
x2 = np.random.lognormal(np.log(gamma_ref_h),1,size=nwalkers)

p0 = np.vstack([x0,x1,x2]).T

# --- Run MCMC sampling --- #
sampler = emcee.EnsembleSampler(nwalkers,ndim,ln_residual_function,args=(strain_list,stress_list))
sampler.run_mcmc(p0,nstep,progress=True)

x0 = sampler.chain[:,:,0].mean(0).T
x1 = sampler.chain[:,:,1].mean(0).T
x2 = sampler.chain[:,:,2].mean(0).T

G0 = np.mean(x0[-int(nstep/4):])
gamma_ref = np.mean(x1[-int(nstep/4):])
gamma_ref_h = np.mean(x2[-int(nstep/4):])


# --- Plot results --- #
print("G0=",G0)
print("gamma_ref=",gamma_ref)
print("gamma_ref_h=",gamma_ref_h)

plt.figure()
plt.subplot(3,1,1)
plt.plot(x0)
plt.hlines([10.e6],0,nstep,"red")
plt.hlines([G0],int(nstep*0.75),nstep,"blue")

plt.subplot(3,1,2)
plt.yscale('log')
plt.plot(x1)
plt.hlines([1.e-3],0,nstep,"red")
plt.hlines([gamma_ref],int(nstep*0.75),nstep,"blue")

plt.subplot(3,1,3)
plt.yscale('log')
plt.plot(x2)
plt.hlines([5.e-3],0,nstep,"red")
plt.hlines([gamma_ref_h],int(nstep*0.75),nstep,"blue")
plt.show()


# x = (1.e6,2.e-3,2.e-3)
# res = scipy.optimize.minimize(residual_function,x,args=(strain_list,stress_list),method="Powell")
# print(res)
