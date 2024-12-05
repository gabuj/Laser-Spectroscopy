import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const
import numpy as numpy
from scipy.signal import find_peaks



#Constants
m_r85=1.4099943e-25#mass of rubidium 85 atom in kg
m_r87=1.4431618e-25#mass of rubidium 87 atom in kg

c= const.c
h= const.h
e= const.e

#calculated values
d=21.8e-2
d_err=0.2e-2
delta_f=c/(4*d)
delta_f_err=delta_f*d_err/d





delta_t=1.6e-6
k=delta_f/delta_t

print(k)