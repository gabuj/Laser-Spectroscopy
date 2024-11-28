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
    #change I0 if you're using
    return I0- (gaussian(x, m1, std1, A1) + gaussian(x, m2, std2, A2) + gaussian(x, m3, std3, A3) + gaussian(x, m4, std4, A4))

def TemperatureRb(mean, std,std_err,m):
    #find temperature using doppler boradening formula
    #convert mean from nm to m
    mean = mean*1e-9
    std = std*1e-9
    std_err = std_err*1e-9
    
    T= m*c**2*std**2/(k*mean**2)
    T_err= T*2*std_err/std
    return T, T_err

def std_from_T(T, T_err, mean, m):
    #find standard deviation from temperature
    std = np.sqrt(k*T*mean**2/(m*c**2))
    std_err = std*T_err/T
    return std, std_err


def calibrate_to_wavelength(peaks, calibration_wavelengths):
    #find distance in time between peaks
    distances = np.diff(peaks)
    #find distance in wavelength between peaks
    wavelength_distances = np.diff(calibration_wavelengths)
    conversion_list=[]
    for i in range(len(distances)):
        conversion_list.append(wavelength_distances[i]/distances[i])
    conversion= np.mean(conversion_list)
    conversion_err= np.std(conversion_list)
    return conversion, conversion_err



#load data
filename = 'doppler_broadened/SP2_26112024.CSV'

data = np.loadtxt(filename, delimiter = ',', skiprows = 1)
t = data[:,0]
intensities = data[:,1]

#plot data
plt.plot(t, intensities, 'b-', label = 'data')
plt.xlabel('time (microseconds?)')
plt.ylabel('Intensity (V)')
plt.legend()
plt.title('Absorption spectrum of rubidium')
plt.show()

#we expect 4 "peaks" (abortions) in the data, so we will fit 4 gaussians to the data

#initial guesses for the fit parameters

m1_guess = 0.303
std1_guess = 0.001
A1_guess = 1.04

m2_guess = 0.309
std2_guess = 0.001
A2_guess = 2.04

m3_guess = 0.3206
std3_guess = 0.001
A3_guess = 9.98

m4_guess = 0.3305
std4_guess = 0.001
A4_guess = 0.26

I0_guess = 0.9

#FIT DATA TO SUM OF 4 GAUSSIANS

initial_guesses = [m1_guess, std1_guess, A1_guess, m2_guess, std2_guess, A2_guess, m3_guess, std3_guess, A3_guess, m4_guess, std4_guess, A4_guess, I0_guess]
#fit gaussians to data
popt, pcov = curve_fit(spectrum, t, intensities, p0 = initial_guesses)


plt.plot(t,intensities, 'b-', label = 'data')
plt.plot(t, spectrum(t, *popt), 'r-', label = 'fit gaussians')
plt.xlabel('time (s)')
plt.ylabel('Intensity (V)')
plt.legend()
plt.title('Absorption spectrum of rubidium with fit')
plt.show()


#FIT EACH GAUSSIAN SEPARATELY. Gaussian 1 between t=0.2989 and t=0.30575, Gaussian 2 between t=0.30575 and t=0.3117, Gaussian 3 between t=0.3181 and t=0.3233, Gaussian 4 between t=0.3282 and t=0.3331

# #Gaussian 1
# indexes_1= (t>0.2989) & (t<0.30575)
# #get first value over threoshold to shift the t_1 values
# t_1 = t[indexes_1]
# intensities_1 = intensities[indexes_1]


# initial_guesses_1 = [m1_guess, std1_guess, A1_guess, I0_guess]
# popt_1, pcov_1 = curve_fit(gaussian, t_1, intensities_1, p0 = initial_guesses_1)

# #Gaussian 2
# indexes_2= (t>0.30575) & (t<0.3117)
# t_2 = t[indexes_2]
# intensities_2 = intensities[indexes_2]


# initial_guesses_2 = [m2_guess, std2_guess, A2_guess, I0_guess]
# popt_2, pcov_2 = curve_fit(gaussian, t_2, intensities_2, p0 = initial_guesses_2)

# #Gaussian 3
# indexes_3= (t>0.3181) & (t<0.3233)
# t_3 = t[indexes_3]
# intensities_3 = intensities[indexes_3]

# initial_guesses_3 = [m3_guess, std3_guess, A3_guess, I0_guess]
# popt_3, pcov_3 = curve_fit(gaussian, t_3, intensities_3, p0 = initial_guesses_3)

# #Gaussian 4
# indexes_4= (t>0.3282) & (t<0.33331)
# t_4 = t[indexes_4]
# intensities_4 = intensities[indexes_4]


# initial_guesses_4 = [m4_guess, std4_guess, A4_guess, I0_guess]
# popt_4, pcov_4 = curve_fit(gaussian, t_4, intensities_4, p0 = initial_guesses_4)


# #plot the fit
# plt.plot(t,intensities, 'b-', label = 'data')
# plt.plot(t, gaussian(t, *popt_1), 'r-', label = 'fit gaussian1')
# plt.plot(t, gaussian(t, *popt_2), 'g-', label = 'fit gaussian2')
# plt.plot(t, gaussian(t, *popt_3), 'y-', label = 'fit gaussian3')
# plt.plot(t, gaussian(t, *popt_4), 'm-', label = 'fit gaussian4')
# plt.xlabel('time (s)')
# plt.ylabel('Intensity (V)')
# plt.legend()
# plt.title('Absorption spectrum of rubidium with fit')
# plt.show()


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
wavelength1 = 780.236 #Rb87_F2
wavelength2 = 780.239 #Rb85_F3
wavelength3 = 780.244 #Rb85_F2
wavelength4 = 780.250 #Rb87_F1

peaks = [first_peak, second_peak, third_peak, fourth_peak]
calibration_wavelengths= [wavelength1, wavelength2, wavelength3, wavelength4]

conversion, conversion_err = calibrate_to_wavelength(peaks, calibration_wavelengths)
#find temperature

wavelength_std_1= conversion*std1
wavelength_std_1_err= conversion_err*std1


wavelength_std_2= conversion*std2
wavelength_std_2_err= conversion_err*std2


wavelength_std_3= conversion*std3
wavelength_std_3_err= conversion_err*std3

wavelength_std_4= conversion*std4
wavelength_std_4_err= conversion_err*std4

T87_F2,T87_F2_err = TemperatureRb(wavelength1, wavelength_std_1,wavelength_std_1_err, m_r87) #first peak is T87_F2 transition

T85_F3,T85_F3_err = TemperatureRb(wavelength2, wavelength_std_2,wavelength_std_2_err, m_r85) #second peak is T85_F3 transition

T85_F2,T85_F2_err = TemperatureRb(wavelength3, wavelength_std_3,wavelength_std_3_err, m_r85) #third peak is T85_F2 transition

T87_F1,T87_F1_err = TemperatureRb(wavelength4, wavelength_std_4,wavelength_std_4_err, m_r87) #fourth peak is T87_F1 transition

print('Temperature of 87Rb F=2 transition: ', T87_F2, '+/-', T87_F2_err, 'K')
print('Temperature of 85Rb F=3 transition: ', T85_F3, '+/-', T85_F3_err, 'K')
print('Temperature of 85Rb F=2 transition: ', T85_F2, '+/-', T85_F2_err, 'K')
print('Temperature of 87Rb F=1 transition: ', T87_F1, '+/-', T87_F1_err, 'K')


#evaluate the expected stds from ambient temperature

T_ambient = 300
T_ambient_err = 1

std_87_F2, std_87_F2_err = std_from_T(T_ambient, T_ambient_err, wavelength1, m_r87)
std_85_F3, std_85_F3_err = std_from_T(T_ambient, T_ambient_err, wavelength2, m_r85)
std_85_F2, std_85_F2_err = std_from_T(T_ambient, T_ambient_err, wavelength3, m_r85)
std_87_F1, std_87_F1_err = std_from_T(T_ambient, T_ambient_err, wavelength4, m_r87)

print('Expected standard deviation of 87Rb F=2 transition: ', std_87_F2, '+/-', std_87_F2_err, 'nm')
print('Found standard deviation of 87Rb F=2 transition: ', wavelength_std_1, '+/-', wavelength_std_1_err, 'nm')

print('Expected standard deviation of 85Rb F=3 transition: ', std_85_F3, '+/-', std_85_F3_err, 'nm')
print('Found standard deviation of 85Rb F=3 transition: ', wavelength_std_2, '+/-', wavelength_std_2_err, 'nm')

print('Expected standard deviation of 85Rb F=2 transition: ', std_85_F2, '+/-', std_85_F2_err, 'nm')
print('Found standard deviation of 85Rb F=2 transition: ', wavelength_std_3, '+/-', wavelength_std_3_err, 'nm')

print('Expected standard deviation of 87Rb F=1 transition: ', std_87_F1, '+/-', std_87_F1_err, 'nm')
print('Found standard deviation of 87Rb F=1 transition: ', wavelength_std_4, '+/-', wavelength_std_4_err, 'nm')



