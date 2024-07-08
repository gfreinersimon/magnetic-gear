import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
import math


def parse_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    lines_to_keep = lines[3:4] + lines[6:]
    # Join the lines back into a single string
    file_content = ''.join(lines_to_keep)
    # Replace commas with dots for decimal points
    file_content = file_content.replace(',', '.')

    # Use pandas to read the content
    from io import StringIO
    data = pd.read_csv(StringIO(file_content), delimiter=r"\s+|\s*,\s*", engine='python')

    # Extract each column to a numpy array
    force = data['Kraft'].to_numpy()
    distance = data['distance'].to_numpy()
    
    return distance, force

z = math.e

def func(x, k,n):
    return k/(x**n)

file_path = "experiment/MAG FORCE ATTRACTION.csv"
file_is_csv = True
if file_is_csv:
    df = pd.read_csv(file_path)

    Force_y = df['Force (N)'].to_numpy()
    Distance_x = df['Distance (m)'].to_numpy()
    Force_y = Force_y * -1
    plt.plot(Distance_x,Force_y)
    plt.show()

    

else:
    Distance_x,Force_y = parse_file(file_path)
Force_y = Force_y * -1


#rectify force:

#rectify distance:
magnet_radius = 7.5e-3
Distance_x = Distance_x# * 1e-2
Distance_x = Distance_x - np.min(Distance_x)+2*magnet_radius


plt.plot(Distance_x, Force_y, 'bo', label='experimental data')

initial_guess = [6.42E-06,2.1]


popt, pcov = curve_fit(func, Distance_x, Force_y)#, initial_guess)
print(popt)

x_fit = np.arange(np.min(Distance_x), np.max(Distance_x), 0.0001)

plt.plot(x_fit, func(x_fit,popt[0],popt[1]), 'r', label='fit params= k=%5.3f n=%5.3f' % tuple(popt))

plt.xlabel('Distance')
plt.ylabel('Force')
plt.legend()
plt.show()