import objects
import numpy as np
from matplotlib import pyplot as plt
import animate
from scipy.integrate import solve_ivp

spinner1 = objects.spinner(2,0.05,1,np.array([-0.0725,0]),0,0,False)
spinner2 = objects.spinner(2,0.05,1,np.array([0.0725,0]),0,10,True)


spinners = [spinner1, spinner2]


dt = 0.001
t_end = 10.0

def get_d(P1,P2):
    return np.sqrt((P1[0]-P2[0])**2+(P1[1]-P2[1])**2)

def mag_F_magnetic(d):
    return 1. * d**-4

t = 0.
ts = [0.]

while(t<=t_end):
    t+=dt
    ts.append(t)
    T = 0
    for i in range(spinner1.n_arms):
        Fi = np.zeros(2)
        for j in range(spinner2.n_arms):
            s1mi = spinner1.get_mi(i)
            s2mj = spinner2.get_mi(j)
            d = get_d(s1mi,s2mj)
            Fi+=mag_F_magnetic(d) * (s2mj - s1mi)/d
        T += np.cross(spinner1.get_ri(i),Fi)
    
    spinner1.dthetas.append(spinner1.dthetas[-1]+dt*T/spinner1.I)
    spinner1.thetas.append(spinner1.thetas[-1] + dt*spinner1.dthetas[-1])

    spinner2.dthetas.append(spinner2.dthetas[-1])
    spinner2.thetas.append(spinner2.thetas[-1]+spinner2.dthetas[-1]*dt)

plt.plot(ts,spinner1.thetas)
plt.plot(ts,spinner2.thetas)
plt.show()

animate.animate(spinner1,spinner2,0.01,800,400,1000.)



        










