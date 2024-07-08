import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


def animate_spinner(sfx,sfy,smx,smy,radius,xlim,ylim):
    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)

    # Plot the initial circles
    sf1 = plt.Circle((sfx[0][0],sfy[0][0]), radius, color='r', fill=False)
    sf2 = plt.Circle((sfx[1][0],sfy[1][0]), radius, color='g', fill=False)
    sf3 = plt.Circle((sfx[2][0],sfy[2][0]), radius, color='b', fill=False)

    sm1 = plt.Circle((smx[0][0],smy[0][0]), radius, color='r', fill=False)
    sm2 = plt.Circle((smx[1][0],smy[1][0]), radius, color='g', fill=False)
    sm3 = plt.Circle((smx[2][0],smy[2][0]), radius, color='b', fill=False)

    ax.add_patch(sf1)
    ax.add_patch(sf2)
    ax.add_patch(sf3)
    ax.add_patch(sm1)
    ax.add_patch(sm2)
    ax.add_patch(sm3)

    # Update function for the animation
    def update(frame):
        
        sm1.set_center((smx[0][frame], smy[0][frame]))
        sm2.set_center((smx[1][frame], smy[1][frame]))
        sm3.set_center((smx[2][frame], smy[2][frame]))
        sf1.set_center((sfx[0][frame], sfy[0][frame]))
        sf2.set_center((sfx[1][frame], sfy[1][frame]))
        sf3.set_center((sfx[2][frame], sfy[2][frame]))
        
        return sm1,sm2,sm3,sf1,sf2,sf3

    # Create the animation
    ani = FuncAnimation(fig, update, frames=np.arange(0,len(smx[0])), blit=True)
    #FFWriter = FFMpegWriter(fps=10)
    #ani.save('animation.mp4',writer=FFWriter)

    

    plt.show()




#readout all necessary data

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

config = load_config('simulation_final/config.json')



output_path = config['out_path']
omega = config['omega']

df_s2 = pd.read_csv(output_path)
ts = df_s2['time[s]'].to_numpy()
thetas = df_s2['theta[rad]'].to_numpy()
phis = config['phi_0']+omega*ts


center_sm = np.array(config['center_sm'])
center_sf = np.array(config['center_sf'])

spinner_radius = config['spinner_radius']
magnet_radius = config['magnet_radius']

smx = []
smy = []
sfx = []
sfy = []

def get_nthm(alpha,n,center,spinner_radius):
    alpha_c  = alpha+n*2./3.*np.pi
    res_x = center[0]+spinner_radius*np.cos(alpha_c)
    res_y = center[1]+spinner_radius*np.sin(alpha_c)
    return res_x,res_y

for i in range(3):
    dsmx,dsmy = get_nthm(phis,i,center_sm,spinner_radius)
    smx.append(dsmx)
    smy.append(dsmy)
    dsfx,dsfy = get_nthm(thetas,i,center_sf,spinner_radius)
    sfx.append(dsfx)
    sfy.append(dsfy)
    #sm.append(spinner_radius*np.array([center_sm[0]+np.cos(phis+2/3*np.pi*i),center_sm[1]+np.sin(phis+2/3*np.pi*i)]))
    #sf.append(spinner_radius*np.array([center_sf[0]+np.cos(thetas+2/3*np.pi*i),center_sf[1]+np.sin(thetas+2/3*np.pi*i)]))

print(len(smx[0]))
lim = config['center_distance']+magnet_radius
animate_spinner(sfx,sfy,smx,smy,magnet_radius,lim*2,lim)





