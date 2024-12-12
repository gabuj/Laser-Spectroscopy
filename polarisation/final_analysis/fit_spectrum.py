import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def lorentzian(x, gamma, x0,A):
    return A*(2/(np.pi)*gamma) * ((gamma)**2)/(4*(x-x0)**2+(gamma)**2)*A

def spectrum(x, gamma_1, x0_1, gamma_2, x0_2, gamma_3, x0_3, gamma_4, x0_4, gamma_5, x0_5, gamma_6, x0_6,I0, A1, A2, A3, A4, A5, A6):
    #change I0 if you're using
    return I0+ (lorentzian(x, gamma_1, x0_1,A1)+lorentzian(x, gamma_2, x0_2,A2)+lorentzian(x, gamma_3, x0_3,A3)+lorentzian(x, gamma_4, x0_4,A4)+lorentzian(x, gamma_5, x0_5,A5)+lorentzian(x, gamma_6, x0_6,A6))

I0_guess= -0.01

# gamma1_guess= 1e7
# x0_1_guess = 6.6e8
# A1_guess = 10e-5

# gamma2_guess = 1e7
# x0_2_guess = 7.3e8
# A2_guess = 18e-5

# gamma3_guess = 1e7
# x0_3_guess = 7.6e8
# A3_guess =  14e-5

# gamma4_guess = 1e7
# x0_4_guess = 8.1e8
# A4_guess =  13e-5

# gamma5_guess = 1e7
# x0_5_guess = 8.5e8
# A5_guess = 6e-5

# gamma6_guess = 1e7
# x0_6_guess = 8.9e8
# A6_guess = 5e-5

gamma1_guess= 1e7
x0_1_guess = 6.6e8
A1_guess = 10e-5

gamma2_guess = 1e7
x0_2_guess = 7.3e8
A2_guess = 18e-5

gamma3_guess = 1e7
x0_3_guess = 7.6e8
A3_guess =  14e-5

gamma4_guess = 1e7
x0_4_guess = 8.1e8
A4_guess =  13e-5

gamma5_guess = 1e7
x0_5_guess = 8.5e8
A5_guess = 6e-5

gamma6_guess = 1e7
x0_6_guess = 8.9e8
A6_guess = 5e-5


initial_guesses = [gamma1_guess, x0_1_guess, gamma2_guess, x0_2_guess, gamma3_guess, x0_3_guess, gamma4_guess, x0_4_guess, gamma5_guess, x0_5_guess, gamma6_guess, x0_6_guess, I0_guess, A1_guess, A2_guess, A3_guess, A4_guess, A5_guess, A6_guess]

def fit(f, intensities):
    #fit gaussians to data
    plt.plot(f,intensities, 'b-', label = 'data')
    #plot guesses
    plt.plot(f, spectrum(f, *initial_guesses), 'g-', label = 'initial guess')
    plt.xlabel('frequency (GHz)')
    plt.ylabel('Intensity (V)')
    plt.legend()
    plt.title('polarisation spectroscopy of rubidium sigma minus pump')
    plt.show()
    
    popt, pcov = curve_fit(spectrum, f, intensities, p0 = initial_guesses)
    plt.plot(f,intensities, 'b-', label = 'data')
    plt.plot(f, spectrum(f, *popt), 'r-', label = 'fit lorentzian')
    #plot guesses
    # plt.plot(f, spectrum(f, *initial_guesses), 'g-', label = 'initial guess')
    plt.xlabel('frequency (GHz)')
    plt.ylabel('Intensity (V)')
    plt.legend()
    plt.title('polarisation spectroscopy of rubidium sigma minus pump')
    plt.show()



