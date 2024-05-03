import numpy as np


class spinner:
    def __init__(self, n_arms, R_magnets, I, center, initial_theta = 0, initial_dtheta = 0, is_forced = False):
        self.n_arms = n_arms
        self.R_magnets = R_magnets
        self.I = I
        self.center = center
        self.angle = 2. * np.pi / n_arms
        self.thetas = [initial_theta]
        self.dthetas =[initial_dtheta]

    def get_ri(self,i):
        ri = self.R_magnets * np.array([np.cos(self.thetas[-1]+i*self.angle),np.sin(self.thetas[-1]+i*self.angle)])
        return ri
    def get_mi(self,i):
        return self.get_ri(i)+self.center
    def get_nthmi(self,n,i):
        mi = self.center + self.R_magnets * np.array([np.cos(self.thetas[n]+i*self.angle),np.sin(self.thetas[n]+i*self.angle)])
        return mi
    def get_ri_theta(self, i, theta):
        ri = self.R_magnets * np.array([np.cos(theta+i*self.angle),np.sin(theta+i*self.angle)])
        return ri
    def get_mi_theta(self, i, theta):
        mi = self.center + self.get_ri_theta(i,theta)
        return mi


