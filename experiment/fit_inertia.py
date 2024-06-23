import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt

#constants:

g = 9.81  #gravitational acceleration [m/s^2]
r = 24e-3 #radius of the wheel [m]
m = 50e-3 #kg

file_path = "experiment/trial_2_inertia.csv"
cutoff = 40

df = pd.read_csv(file_path)

thetas = -df['Latest: Angle (rad)']
omegas = -df['Latest: Velocity (rad/s)']

angle = 0
index = 0

while(angle < cutoff):
    angle = thetas[index]
    index+=1

thetas = thetas[:index]
omegas = omegas[:index]


def func(theta,I):
    res = 2*m*g*r/(m*r**2+I)*theta
    return res

I_res, _ = curve_fit(func,thetas,omegas**2)
print(f'The moment of Inertia is {I_res}')

plt.plot(thetas,omegas**2,label='data')
plt.plot(thetas,func(thetas,I_res),label='fit')
plt.legend()
plt.show()