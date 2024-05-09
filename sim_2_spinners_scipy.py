import objects
import numpy as np
from matplotlib import pyplot as plt
import animate
from scipy.integrate import solve_ivp

#spinner(number of magnets, radius, moment of inertia, position of the centers, - , - ,-)
spinner1 = objects.spinner(3,0.05,1,np.array([-0.0725,0]),0,0,False)
spinner2 = objects.spinner(3,0.05,1,np.array([0.0725,0]),0,1,True)

#inital angle and omega for spinner 2
theta2_0 = 0
dtheta2 = 1

#[initial angle, initial omega] spinner 1
y0 = [0,0]

def get_d(P1,P2):
    return np.sqrt((P1[0]-P2[0])**2+(P1[1]-P2[1])**2)

def mag_F_magnetic(d):
    return 1. * d**-4
mu = 0
def ode(t,y):
    theta, dtheta = y
    T = 0
    for i in range(spinner1.n_arms):
        Fi = np.zeros(2)
        for j in range(spinner2.n_arms):
            s1mi = spinner1.get_mi_theta(i,theta)
            s2mj = spinner2.get_mi_theta(j, dtheta2 * t + theta2_0)
            d = get_d(s1mi,s2mj)
            Fi+=mag_F_magnetic(d) * (s2mj - s1mi)/d
        T += np.cross(spinner1.get_ri_theta(i,theta),Fi) 
    return [dtheta, T/spinner1.I]

t_span = (0,10)

T_thetas = np.linspace(0,2*np.pi,1000)
T_ys = np.zeros((2,1000))
T_ys[0,:] = T_thetas
Ts = ode(T_thetas, )

ts_eval = np.linspace(0,10,1000)

sol = solve_ivp(ode, t_span, y0, t_eval = ts_eval,method = 'Radau' )

spinner1.thetas, spinner1.dthetas = sol.y
spinner2.thetas = ts_eval * dtheta2 + theta2_0

plt.plot(sol.t, sol.y[0])
plt.show()



plt.plot(ts_eval,spinner1.thetas)
plt.plot(ts_eval,spinner2.thetas)
plt.show()

plt.plot(np.mod(sol.y[0],2/3*np.pi),sol.y[1])
plt.show()

animate.animate(spinner1,spinner2,0.1,800,400,1000.)



        










