import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
import random

#setup parameters:


spinner_radius = 3e-2 #distance from center of spinner to center of magnet[m]
center_distance = 8.5e-2 #distance between centers of spinners[m]
magnetic_force_constant = 5.2e-07 #determined from fitting[N*m^exponent]
exponent = 3.62 #determined from the fit as well dimensionless
omega = -2.35 #angular velocity of motorized spinner

orientation = 1 # 1 or -1 if 1 magnets repell if -1 magnets attract

center_sm = [0,0]
center_sf = [center_distance, 0]


I_spinner = 8.24885069e-05 #determined from fitting inertia curves [kg*m^2]
m_magnet = 2.69e-3 #mass of one magnet [kg]
n_magnets = 2 #number of magnets per stack
magnet_radius = 0.0075 #radius of the magnets [m]
I_total = I_spinner + 3*n_magnets*m_magnet*(1./2.*magnet_radius**2+spinner_radius**2)


#initial conditons
theta_0 = 0 #magnet pointing away from center of other spinner [rad]
theta_dot_0 = 0 # positive: counter clockwise [rad/s]
phi_0 = 0 #initial angle of motorized spinner if 0 magnet points towards center of other spinner 

#output
output_path = 'out.csv'

#helper functions
def calc_magnetic_force_magnitude(d_magnet_center):
    res = orientation * magnetic_force_constant/d_magnet_center**exponent
    return res

def calc_rel_pos_m(angle,m_index):
    angle = angle + 2./3. * np.pi * m_index
    pos_m = np.array([np.cos(angle),np.sin(angle)])*spinner_radius
    return pos_m

def calc_sum_torques(theta,phi):
    sum_torque = 0

    for m_sf in range(3): #m_sf 0,1,2 corresponding to the three different magnets
        r_vec = calc_rel_pos_m(theta,m_sf) #radius vector connecting the center of free spinner to the magnet m_sf
        pos_m_sf = center_sf + r_vec #position vector of free spinner magnet m_sf
        
        for m_sm in range(3):
            pos_m_sm = center_sm + calc_rel_pos_m(phi,m_sm)
            connecting_vector = pos_m_sm - pos_m_sf #vector pointing from motorized spinner magnet to free spinner magnet
            distance = np.sqrt(connecting_vector[0]**2 + connecting_vector[1]**2)
            connecting_unit_vector = connecting_vector/distance
            force = calc_magnetic_force_magnitude(distance)*connecting_unit_vector
            torque = np.cross(r_vec,force)
            sum_torque += torque #add calculated torque to the sum of torques

    return sum_torque

            
#state = [theta,theta_dot]

def diff_eq(t,state):
    theta = state[0]
    thetap = state[1]
    phi = phi_0 + omega * t
    sum_torques = calc_sum_torques(theta,phi)-thetap*0.00001
    thetapp = sum_torques/(I_total)
    dstate = [thetap,thetapp]
    return dstate


#simulation
final_time = 60 #time where simulation stops[s]

timespan = (0,final_time)

n_eval = 1000 #number of times solution is saved
dt_eval = final_time/n_eval

evals = np.linspace(0,final_time, n_eval)

#varying parameters:

dist_range = np.linspace(7e-2,0.1,10)
omega_range = np.linspace(-5,-1,10)

max_angles = []
std_devs = []
for d in omega_range:
    max_angles_d = []
    for i in range (10):
        print(f"iteration {i} at omega {d}")

        theta_0 = theta_0 + (random.random()-0.5)*0.1
        init_state = [theta_0,d]
        solution = solve_ivp(diff_eq, timespan, init_state, method='RK45',t_eval=evals)
        ts = solution.t
        thetas = solution.y[0]
        thetaps = solution.y[1]
        max_angles_d.append(np.max(thetas))
    max_angles.append(np.average(max_angles_d))    
    std_devs.append(np.std(max_angles_d))

output_path_variation = 'out_variation.csv'
df_variation = pd.DataFrame({
    'distances[m]': omega_range,
    'theta max[rad]': max_angles,
    'std dev[rad]': std_devs
})

df_variation.to_csv(output_path_variation, index=False)



plt.scatter(omega_range,max_angles,c='r')
plt.errorbar(omega_range,max_angles,std_devs,capsize=20,c='black')
plt.show()

'''
phis = phi_0 + ts * omega

torques = []
for i in range(len(phis)):
    torques.append(calc_sum_torques(thetas[i],phis[i]))

plt.plot(ts,thetaps)
plt.show()
plt.plot(ts,thetas-np.pi)
plt.show()


#save data:

df = pd.DataFrame({
    'time[s]': ts,
    'theta[rad]': thetas,
    'thetap[rad/s]': thetaps
})

df.to_csv(output_path, index=False)

print(f'time step = {dt_eval}')
print(f'center distance = {center_distance}')

'''