import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import sys

class Cyclic_HD():
    def __init__(self,G0,gamma_ref=1.e-3,gamma_ref_h=None,Yoshida=False):
        # HD parameters
        self.G0 = G0
        self.gr = gamma_ref
        self.Yoshida = Yoshida

        # Ishihara Yoshida model
        if Yoshida:
            self.G0_h = G0
            self.gr_h = gamma_ref_h
        else:
            self.G0_h = G0
            self.gr_h = gamma_ref

        # reversal point
        self.tau0 = 0.0
        self.gamma0 = 0.0

        self.tau0_p_list = []
        self.gamma0_p_list = []

        self.tau0_n_list = []
        self.gamma0_n_list = []

        # yield stress
        self.tau_y = 0.0

    # -------------------------------------------------------------------------------------- #
    def skeleton_curve(self,gamma):
        tau = self.G0 * gamma / (1+np.abs(gamma)/self.gr)
        return tau

    def hysteresis_curve(self,gamma):
        def skeleton_curve_for_hyst(gamma):
            tau = self.G0_h * gamma / (1+np.abs(gamma)/self.gr_h)
            return tau

        tau = self.tau0 + 2*skeleton_curve_for_hyst(0.5*(gamma-self.gamma0))
        return tau

    def update_yield_stress(self,gamma,tau):
        self.tau_y = np.abs(tau)
        if self.Yoshida:
            if np.abs(gamma) > 0.0:
                self.G0_h = abs(tau/gamma) * (1+np.abs(gamma)/self.gr_h)

    # -------------------------------------------------------------------------------------- #
    def check_reversal(self,gamma,g0,tau0,dg):
        if dg*(gamma-g0) < 0.0:
            self.tau0 = tau0
            self.gamma0 = g0
            self.update_reversal_points(dg)

    def update_reversal_points(self,dg):
        tau0_list = [self.tau0]
        gamma0_list = [self.gamma0]

        if dg >= 0.0:
            for i,tau0 in enumerate(self.tau0_p_list):
                if tau0 > self.tau0:
                    tau0_list = [self.tau0] + self.tau0_p_list[i:]
                    gamma0_list = [self.gamma0] + self.gamma0_p_list[i:]
                    break
            self.tau0_p_list = tau0_list
            self.gamma0_p_list = gamma0_list
            # print("p",self.tau0_p_list,self.gamma0_p_list)

        else:
            for i,tau0 in enumerate(self.tau0_n_list):
                if tau0 < self.tau0:
                    tau0_list = [self.tau0] + self.tau0_n_list[i:]
                    gamma0_list = [self.gamma0] + self.gamma0_n_list[i:]
                    break
            self.tau0_n_list = tau0_list
            self.gamma0_n_list = gamma0_list
            # print("n",self.tau0_n_list,self.gamma0_n_list)

    def find_hysteresis_curve(self,gamma,dg):
        tau = self.hysteresis_curve(gamma)
        if np.abs(tau) >= self.tau_y:
            tau = self.skeleton_curve(gamma)
            self.update_yield_stress(gamma,tau)
            return tau

        if len(self.tau0_p_list) == 0:
            return tau
        if len(self.tau0_n_list) == 0:
            return tau

        if dg >= 0.0:
            if tau >= self.tau0_p_list[0]:
                self.tau0_p_list.pop(0)
                self.gamma0_p_list.pop(0)

                self.tau0 = self.tau0_n_list.pop(0)
                self.gamma0 = self.gamma0_n_list.pop(0)
                # print("pop positive",self.tau0,self.gamma0)

                tau = self.hysteresis_curve(gamma)
                return tau
        else:
            if tau <= self.tau0_n_list[0]:
                self.tau0_n_list.pop(0)
                self.gamma0_n_list.pop(0)

                self.tau0 = self.tau0_p_list.pop(0)
                self.gamma0 = self.gamma0_p_list.pop(0)
                # print("pop negative",self.tau0,self.gamma0)

                tau = self.hysteresis_curve(gamma)
                return tau

        return tau

    # -------------------------------------------------------------------------------------- #
    def cyclic_shear(self,shear_strain,plot=False):
        # print("+++ cyclic shear +++")
        nstep = len(shear_strain)

        gamma_list,tau_list = [],[]
        g0,tau0 = 0.0,0.0
        dg = 0.0
        for i in range(nstep):
            gamma = shear_strain[i]

            self.check_reversal(gamma,g0,tau0,dg)
            tau = self.find_hysteresis_curve(gamma,dg)

            gamma_list += [gamma]
            tau_list += [tau]

            dg = gamma-g0
            g0 = gamma
            tau0 = tau

            # print(gamma,tau,self.gamma0,self.tau0,self.tau_y)

        if plot:
            plt.figure()
            plt.plot(gamma_list,tau_list)
            plt.show()

        return np.array(tau_list)
