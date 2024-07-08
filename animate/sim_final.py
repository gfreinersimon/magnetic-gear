import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
import json

# Function to load configuration from a JSON file
def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

# Load configuration
config = load_config('animate/config.json')

# Accessing the variables
spinner_radius = config['spinner_radius']
center_distance = config['center_distance']
magnetic_force_constant = config['magnetic_force_constant']
exponent = config['exponent']
omega = config['omega']
orientation = config['orientation']
center_sm = config['center_sm']
center_sf = config['center_sf']
I_spinner = config['I_spinner']
m_magnet = config['m_magnet']
n_magnets = config['n_magnets']
magnet_radius = config['magnet_radius']
I_total = config['I_total']
theta_0 = config['theta_0']
theta_dot_0 = config['theta_dot_0']
phi_0 = config['phi_0']

output_path = config['out_path']

I_total = I_spinner + 3*n_magnets*m_magnet*(1./2.*magnet_radius**2**2)

#helper functions

max_force = 0.2
min_force = 1e-2
def calc_magnetic_force_magnitude(d_magnet_center):
    res = orientation * magnetic_force_constant/d_magnet_center**exponent
    '''
    if res > max_force:
        print(f"capping force calculated is: {res}")
        res = max_force
    elif res < min_force:
        print(f"capping downwards force calculated is : {res}")
        res = 0#min_force
    '''
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
            connecting_vector = pos_m_sf - pos_m_sm #vector pointing from motorized spinner magnet to free spinner magnet
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
    sum_torques = calc_sum_torques(theta,phi)# - thetap/np.abs(thetap)*1e-5 
    thetapp = sum_torques/(I_total)
    dstate = [thetap,thetapp]
    return dstate


#simulation
final_time = config['final_time'] #time where simulation stops[s]

timespan = (0,final_time)
init_state = [theta_0,theta_dot_0]

n_eval = config['n_eval'] #number of times solution is saved
dt_eval = final_time/n_eval

evals = np.linspace(0,final_time, n_eval)

solution = solve_ivp(diff_eq, timespan, init_state, method='RK45',t_eval=evals)



ts = solution.t
thetas = solution.y[0]
thetaps = solution.y[1]

phis = ts * omega + phi_0
torques = []
for i in range(len(phis)):
    torques.append(calc_sum_torques(phis[i],thetas[i]))

if(False):
    plt.plot(ts,torques)
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
