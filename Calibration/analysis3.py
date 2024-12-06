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

#define function to fit to doppler broadening data

def gaussian(x, m,std,A):
    return A*np.exp(-(x-m)**2/(2*std**2))

#Define Lorentzian function
#Gamma is the full width at half maximum
#X corresponds to the frequency
#X0 corresponds to the central frequency of the transition
def lorentzian(x, gamma, x0,A):
    return A*(2/(np.pi)*gamma) * ((gamma)**2)/(4*(x-x0)**2+(gamma)**2)*A

def spectrum(x, gamma_1, x0_1, gamma_2, x0_2, gamma_3, x0_3, gamma_4, x0_4, gamma_5, x0_5, gamma_6, x0_6, gamma_7, x0_7, gamma_8, x0_8, gamma_9, x0_9, gamma_10, x0_10, gamma_11, x0_11, gamma_12, x0_12, gamma_13, x0_13, gamma_14, x0_14, gamma_15, x0_15, gamma_16, x0_16, gamma_17, x0_17, gamma_18, x0_18, gamma_19, x0_19, gamma_20, x0_20, gamma_21, x0_21, gamma_22, x0_22, gamma_23, x0_23, gamma_24, x0_24, I0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24):
    #change I0 if you're using
    return I0+ (lorentzian(x, gamma_1, x0_1,A1)+lorentzian(x, gamma_2, x0_2,A2)+lorentzian(x, gamma_3, x0_3,A3)+lorentzian(x, gamma_4, x0_4,A4)+lorentzian(x, gamma_5, x0_5,A5)+lorentzian(x, gamma_6, x0_6,A6)+lorentzian(x, gamma_7, x0_7,A7)+lorentzian(x, gamma_8, x0_8,A8)+lorentzian(x, gamma_9, x0_9,A9)+lorentzian(x, gamma_10, x0_10,A10)+lorentzian(x, gamma_11, x0_11,A11)+lorentzian(x, gamma_12, x0_12,A12) + lorentzian(x,gamma_13, x0_13,A13)+lorentzian(x, gamma_14, x0_14,A14)+lorentzian(x, gamma_15, x0_15,A15)+lorentzian(x, gamma_16, x0_16,A16)+lorentzian(x, gamma_17, x0_17,A17)+lorentzian(x, gamma_18, x0_18,A18)+lorentzian(x, gamma_19, x0_19,A19)+lorentzian(x, gamma_20, x0_20,A20)+lorentzian(x, gamma_21, x0_21,A21)+lorentzian(x, gamma_22, x0_22,A22)+lorentzian(x, gamma_23, x0_23,A23)+lorentzian(x, gamma_24, x0_24,A24))

def calibration_fit(t,a,b,c):
    f= a*t**2+b*t+c
    return  f

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
filename = 'Calibration/WF_4.csv'
background_filename = 'Calibration/WFBG_4.csv'
calibration_filename = 'Calibration/WFCAL_4.csv'
data = np.loadtxt(filename, delimiter = ',', skiprows = 1)
background = np.loadtxt(background_filename, delimiter = ',', skiprows = 1)
calibration = np.loadtxt(calibration_filename, delimiter = ',', skiprows = 1)

#extract data
t = np.array(data[:,0])
intensities = np.array(data[:,1])  - np.array(background[:,1])
calibration_t = np.array(calibration[:,0])
calibration_intensities = np.array(calibration[:,1])

t_background = np.array(background[:,0])
intensities_background = np.array(background[:,1])

#make time start from 0
# t = t-t[0]
# calibration_t = calibration_t - calibration_t[0]
# t_background = t_background - t_background[0]



# #plot data
plt.plot(t, intensities, 'b-', label = 'data')
# plt.plot(calibration_t, calibration_intensities, 'r-', label = 'calibration 1')
# plt.plot(t_background, intensities_background, 'g-', label = 'background')
plt.xlabel('time (s)')
plt.ylabel('Intensity (V)')
plt.legend()
plt.title('Doppler free spectrum of rubidium')
plt.show()


#find poeaks and plot
peaks, _ = find_peaks(calibration_intensities, height=0.026, distance=1000)

#plot peaks on graph
plt.plot(calibration_t, calibration_intensities, 'r-', label = 'calibration 1')
plt.plot(calibration_t[peaks], calibration_intensities[peaks], 'x')
plt.xlabel('time (s)')
plt.ylabel('Intensity (V)')
plt.legend()
plt.title('Calibration spectrum of fabry perot')
plt.show()

#find the time of the peaks
peak_times = calibration_t[peaks]
n_peaks= np.arange(0,len(peaks))
peaks_freq= n_peaks*delta_f
peaks_freq_err= delta_f_err*n_peaks
#only get first 13 values
# peak_times = peak_times[:13]
# peaks_freq = peaks_freq[:13]

a_guess= -200
b_guess= 198000000000
c_guess= -14200000000

plt.errorbar(peak_times, peaks_freq, yerr=peaks_freq_err, fmt='x', label = 'calibration')
plt.plot(peak_times, calibration_fit(peak_times, a_guess,b_guess,c_guess), 'r-', label = 'initial guess')
plt.xlabel('mid peak time (s)')
plt.ylabel('distance between peaks (s)')
plt.legend()
plt.title('Calibration spectrum of fabry perot')
plt.show()



#we expect the scanning to not be linear but quadratic, fit the difference in peak (which should be equal) to a line and see the slope. This will be the conversion factor from time to wavelength
popt_cal, pcov_cal = curve_fit(calibration_fit, peak_times, peaks_freq, p0 = [a_guess,b_guess,c_guess], sigma=peaks_freq_err)
a_cal= popt_cal[0]
b_cal= popt_cal[1]
c_cal= popt_cal[2]

a_cal_err= np.sqrt(pcov_cal[0,0])
b_cal_err= np.sqrt(pcov_cal[1,1])
c_cal_err= np.sqrt(pcov_cal[2,2])
 
print(a_cal, b_cal, c_cal)

#plot fit

plt.plot(peak_times, peaks_freq, 'r-', label = 'calibration')
plt.plot(peak_times, calibration_fit(peak_times, a_cal, b_cal,c_cal), 'b-', label = 'fit')
plt.xlabel('mid peak time (s)')
plt.ylabel('distance between peaks (s)')
plt.legend()
plt.title('Calibration spectrum of fabry perot')
plt.show()


#convert from time to frequency assuming f0=0 using delta_f calculated above
f= a_cal*t**2+b_cal*t + c_cal
f_err= np.sqrt((a_cal_err*t**2)**2+(b_cal_err*t)**2+(c_cal_err)**2)

#plot intensities against frequency

plt.plot(f, intensities, 'b-', label = 'data')
plt.xlabel('frequency (Hz)')
plt.ylabel('Intensity (V)')
plt.legend()
plt.title('Doppler free spectrum of rubidium')
plt.show()




#we expect 3 spliting per ground level weith one burn hole each so 6 peals in total times 4 ground states = 24 peaks

#initial guesses for the lorenzian fit parameters

gamma1_guess= 0.0001
x0_1_guess = 0.08000
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

gamma7_guess = 0.0001
x0_7_guess = 0.0915
A7_guess = 50

gamma8_guess = 0.0001
x0_8_guess = 0.0917
A8_guess = 80

gamma9_guess = 0.0001
x0_9_guess = 0.0919
A9_guess = 60

gamma10_guess = 0.0001
x0_10_guess = 0.0920
A10_guess = 30

gamma11_guess = 0.0001
x0_11_guess = 0.0921
A11_guess = 22

gamma12_guess = 0.0001
x0_12_guess = 0.0922
A12_guess = 16

gamma13_guess = 0.0001
x0_13_guess = 0.1042
A13_guess = 60

gamma14_guess = 0.0001
x0_14_guess = 0.1044
A14_guess = 170

gamma15_guess = 0.0001
x0_15_guess = 0.1046
A15_guess = 150

gamma16_guess = 0.0001
x0_16_guess = 0.1048
A16_guess = 55

gamma17_guess = 0.0001
x0_17_guess = 0.1049
A17_guess = 40

gamma18_guess = 0.0001
x0_18_guess = 0.1050
A18_guess = 20

gamma19_guess = 0.00008
x0_19_guess = 0.1093
A19_guess = 50

gamma20_guess = 0.0001
x0_20_guess = 0.1100
A20_guess = 110

gamma21_guess = 0.0001
x0_21_guess = 0.1103
A21_guess =80

gamma22_guess = 0.0001
x0_22_guess = 0.1106
A22_guess = 25

gamma23_guess = 0.0001
x0_23_guess = 0.1110
A23_guess = 23

gamma24_guess = 0.0001
x0_24_guess = 0.1114
A24_guess = 20


I0_guess = -0.02

#FIT DATA TO SUM OF 4 GAUSSIANS

initial_guesses = [gamma1_guess, x0_1_guess, gamma2_guess, x0_2_guess, gamma3_guess, x0_3_guess, gamma4_guess, x0_4_guess, gamma5_guess, x0_5_guess, gamma6_guess, x0_6_guess, gamma7_guess, x0_7_guess, gamma8_guess, x0_8_guess, gamma9_guess, x0_9_guess, gamma10_guess, x0_10_guess, gamma11_guess, x0_11_guess, gamma12_guess, x0_12_guess, gamma13_guess, x0_13_guess, gamma14_guess, x0_14_guess, gamma15_guess, x0_15_guess, gamma16_guess, x0_16_guess, gamma17_guess, x0_17_guess, gamma18_guess, x0_18_guess, gamma19_guess, x0_19_guess, gamma20_guess, x0_20_guess, gamma21_guess, x0_21_guess, gamma22_guess, x0_22_guess, gamma23_guess, x0_23_guess, gamma24_guess, x0_24_guess, I0_guess, A1_guess, A2_guess, A3_guess, A4_guess, A5_guess, A6_guess, A7_guess, A8_guess, A9_guess, A10_guess, A11_guess, A12_guess, A13_guess, A14_guess, A15_guess, A16_guess, A17_guess, A18_guess, A19_guess, A20_guess, A21_guess, A22_guess, A23_guess, A24_guess]
#fit gaussians to data
popt, pcov = curve_fit(spectrum, t, intensities, p0 = initial_guesses)


plt.plot(t,intensities, 'b-', label = 'data')
plt.plot(t, spectrum(t, *popt), 'r-', label = 'fit lorentzian')
#plot guesses
# plt.plot(t, spectrum(t, *initial_guesses), 'g-', label = 'initial guess')
plt.xlabel('time (ms)')
plt.ylabel('Intensity (V)')
plt.legend()
plt.title('Absorption spectrum of rubidium with fit')
plt.show()


#I expect the first peak to be the first 85R ground state transition, the second peak to be the 85R first excited state transition, the third peak to be the 87R ground state transition and the fourth peak to be the 87R first excited state transition
