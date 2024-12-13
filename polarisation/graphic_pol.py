import numpy as np
import matplotlib.pyplot as plt


filename = 'polarisation/all_angles_rotating_linpol_20.txt'
angle, power, power_err = np.loadtxt(filename, usecols=(0, 1,2), skiprows= 2, unpack=True)

angle= angle*np.pi/180

plt.polar(angle, power)
plt.show()