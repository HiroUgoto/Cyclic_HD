import time
import numpy as np
import matplotlib.pyplot as plt
import cyclic_hd

# start = time.time()

# clay = cyclic_hd.Cyclic_HD(G0=10e6,gamma_ref=1.e-3)
clay = cyclic_hd.Cyclic_HD(G0=10e6,gamma_ref=1.e-3,gamma_ref_h=5.e-3,Yoshida=True)

tim = np.linspace(0,10,1000,endpoint=False)
fp = 1.0 # [Hz]
amp = 5.e-3

shear_strain = amp*np.sin(2*np.pi*fp*tim)
shear_stress = clay.cyclic_shear(shear_strain)


#------ Add random noise ---#
std = np.std(shear_stress)

shear_stress_noise = shear_stress + np.random.normal(0.0,0.1*std,size=len(tim))

plt.figure()
plt.plot(tim,shear_stress)
plt.plot(tim,shear_stress_noise)
plt.show()

output_line = np.vstack([shear_strain,shear_stress,shear_stress_noise]).T
np.savetxt("result/test_data_5e3.dat",output_line)


# elapsed_time = time.time() - start
# print ("elapsed_time: {0}".format(elapsed_time) + "[sec]")
