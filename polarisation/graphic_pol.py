import numpy as np
import matplotlib.pyplot as plt


filename = 'polarisation/total_polarisation_circular.txt'
angle, power, power_err = np.loadtxt(filename, usecols=(0, 1,2), skiprows= 1, unpack=True)

angle= angle*np.pi/180

plt.polar(angle, power)
plt.show()