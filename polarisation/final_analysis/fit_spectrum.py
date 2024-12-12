import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def lorentzian(x, gamma, x0,A):
    return A*(2/(np.pi)*gamma) * ((gamma)**2)/(4*(x-x0)**2+(gamma)**2)*A

def spectrum(x, gamma_1, x0_1, gamma_2, x0_2, gamma_3, x0_3, gamma_4, x0_4, gamma_5, x0_5, gamma_6, x0_6, gamma_7, x0_7, gamma_8, x0_8, gamma_9, x0_9, gamma_10, x0_10, gamma_11, x0_11, gamma_12, x0_12, gamma_13, x0_13, gamma_14, x0_14, gamma_15, x0_15, gamma_16, x0_16, gamma_17, x0_17, gamma_18, x0_18, gamma_19, x0_19, gamma_20, x0_20, gamma_21, x0_21, gamma_22, x0_22, gamma_23, x0_23, gamma_24, x0_24, I0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24):
    #change I0 if you're using
    return I0+ (lorentzian(x, gamma_1, x0_1,A1)+lorentzian(x, gamma_2, x0_2,A2)+lorentzian(x, gamma_3, x0_3,A3)+lorentzian(x, gamma_4, x0_4,A4)+lorentzian(x, gamma_5, x0_5,A5)+lorentzian(x, gamma_6, x0_6,A6))


gamma1_guess= 0.0001
x0_1_guess = 6.6e8
A1_guess = 20

gamma2_guess = 0.0001
x0_2_guess = 0.0803
A2_guess = 35

gamma3_guess = 0.0001
x0_3_guess = 0.0805
A3_guess = 28

gamma4_guess = 0.0001
x0_4_guess = 0.0807
A4_guess = 26

gamma5_guess = 0.0001
x0_5_guess = 0.0810
A5_guess = 12

gamma6_guess = 0.0001
x0_6_guess = 0.0812
A6_guess = 12
initial_guesses = [gamma1_guess, x0_1_guess, gamma2_guess, x0_2_guess, gamma3_guess, x0_3_guess, gamma4_guess, x0_4_guess, gamma5_guess, x0_5_guess, gamma6_guess, x0_6_guess]

def fit(f, intensities):
    #fit gaussians to data
    popt, pcov = curve_fit(spectrum, f, intensities, p0 = initial_guesses)
    plt.plot(f,intensities, 'b-', label = 'data')
    plt.plot(f, spectrum(f, *popt), 'r-', label = 'fit lorentzian')
    #plot guesses
    # plt.plot(t, spectrum(t, *initial_guesses), 'g-', label = 'initial guess')
    plt.xlabel('time (ms)')
    plt.ylabel('Intensity (V)')
    plt.legend()
    plt.title('Absorption spectrum of rubidium with fit')
    plt.show()



