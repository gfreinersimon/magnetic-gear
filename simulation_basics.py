from matplotlib import pyplot as plt
import numpy as np

k = 3
x0 = 1
delta_t = 0.0001


ts = [0]
xs = [x0]

t_end = 10
t = 0
xl = x0

while(t<= t_end):
    t = t + delta_t
    x = k * xl * delta_t + xl
    ts.append(t)
    xs.append(x)
    xl = x

ts_np = np.array(ts)
plt.plot(ts,x0*np.exp(k*ts_np))
plt.plot(ts,xs)
plt.show()