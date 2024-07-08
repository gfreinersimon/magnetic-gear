import pandas as pd
import numpy as np

# Path to your text file
file_path = 'experiment/5 new magnets, repulsion.txt'

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
time = data['Zeit'].to_numpy()
force = data['Kraft'].to_numpy()
angle = data['Winkel'].to_numpy()
distance = data['distance'].to_numpy()

# Display the numpy arrays
print("Time array:", time)
print("Force array:", force)
print("Angle array:", angle)
print("Distance array:", distance)
