import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Constants
m_r85=1.4099943e-25#mass of rubidium 85 atom in kg
m_r87=1.4431618e-25#mass of rubidium 87 atom in kg
c=3e8#speed of light in m/s
k=1.38e-23#Boltzmann constant in J/K




#define function to fit to doppler broadening data

def gaussian(x, m,std,A):
    return A*np.exp(-(x-m)**2/(2*std**2))

def spectrum(x, m1, std1, A1, m2, std2, A2, m3, std3, A3, m4, std4, A4, I0):
    return I0- (gaussian(x, m1, std1, A1) + gaussian(x, m2, std2, A2) + gaussian(x, m3, std3, A3) + gaussian(x, m4, std4, A4))

def Temperature85(mean, std,m):
    #find temperature using doppler boradening formula
    return m*c**2*std**2/(k*mean**2)

def calibrate_to_wavelength(x,peaks, calibration_wavelengths):
    #calibrate to wavelength
    slope = (calibration_wavelengths[1]-calibration_wavelengths[0])/(peaks[1]-peaks[0])
    intercept = calibration_wavelengths[0]-slope*peaks[0]
    
    slope2 = (calibration_wavelengths[2]-calibration_wavelengths[1])/(peaks[2]-peaks[1])
    intercept2 = calibration_wavelengths[1]-slope3*peaks[1]

    slope3 = (calibration_wavelengths[3]-calibration_wavelengths[2])/(peaks[3]-peaks[2])
    intercept3 = calibration_wavelengths[2]-slope2*peaks[2]
       
    meanslope = (slope+slope2+slope3)/4
    meanintercept = (intercept+intercept2+intercept3)/4
    
    sloap_std = np.std([slope,slope2,slope3])

    wavelengths= meanslope*x+meanintercept
    
    wavelength_uncertainties= np.sqrt((sloap_std*x)**2+(meanslope*(x[1]-x[0]))**2+(meanintercept)**2)
    
    return meanslope, meanintercept, wavelengths, wavelength_uncertainties



#load data
filename = 'doppler_broadened/SP2_22112024.CSV'

data = np.loadtxt(filename, delimiter = ',', skiprows = 1)
t = data[:,0]
intensities = data[:,1]

#plot data
plt.plot(t, intensities, 'b-', label = 'data')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (V)')
plt.legend()
plt.title('Absorption spectrum of rubidium')

#we expect 4 "peaks" (abortions) in the data, so we will fit 4 gaussians to the data

#initial guesses for the fit parameters

m1_guess = 780
std1_guess = 1
A1_guess = -0.5

m2_guess = 780.5
std2_guess = 1
A2_guess = -0.5

m3_guess = 781
std3_guess = 1
A3_guess = -0.5

m4_guess = 781.5
std4_guess = 1
A4_guess = -0.5
I0_guess = np.max(intensities)

initial_guesses = [m1_guess, std1_guess, A1_guess, m2_guess, std2_guess, A2_guess, m3_guess, std3_guess, A3_guess, m4_guess, std4_guess, A4_guess, I0_guess]
#fit gaussians to data
popt, pcov = curve_fit(spectrum, t, intensities, p0 = initial_guesses)

#plot the fit
plt.plot(t,intensities, 'b-', label = 'data')
plt.plot(t, spectrum(t, *popt), 'r-', label = 'fit')
plt.plot(t, spectrum(t, spectrum(t, *initial_guesses)), 'g-', label = 'initial guess')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (V)')
plt.legend()
plt.title('Absorption spectrum of rubidium with fit')
plt.show()


#I expect the first peak to be the first 85R ground state transition, the second peak to be the 85R first excited state transition, the third peak to be the 87R ground state transition and the fourth peak to be the 87R first excited state transition
first_peak= popt[0]
std1 = popt[1]

second_peak = popt[3]
std2 = popt[4]

third_peak = popt[6]
std3 = popt[7]

fourth_peak = popt[9]
std4 = popt[10]

#map from t to wavelength
wavelength1 = 780.24
wavelength2 = 782
wavelength3 = 783
wavelength4 = 785

peaks = [first_peak, second_peak, third_peak, fourth_peak]
calibration_wavelengths= [wavelength1, wavelength2, wavelength3, wavelength4]


#find temperature

mean = first_peak
std = std1
T85_1 = Temperature85(wavelength1, std, m_r85) #first peak is 85R ground state transition