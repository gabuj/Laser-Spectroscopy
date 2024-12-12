import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const
from scipy.signal import find_peaks
#constants
c=const.c

d=21.8e-2
d_err=0.2e-2
delta_f=c/(4*d)
delta_f_err=delta_f*d_err/d

def calibration_fit(t,a,b,c):
    f= a*t**2+b*t+c
    return  f

def calibrate(calibration_t, calibration_intensities):
    #calibrate data
    #find peaks
    peaks, _ = find_peaks(calibration_intensities, height=0.02, distance=1000)
    
    #plot peaks on graph
    plt.plot(calibration_t, calibration_intensities, 'r-', label = 'calibration 1')
    plt.plot(calibration_t[peaks], calibration_intensities[peaks], 'x')
    plt.show()

    #find the time of the peaks
    peak_times = calibration_t[peaks]
    n_peaks= np.arange(0,len(peaks))
    peaks_freq= n_peaks*delta_f
    peaks_freq_err= delta_f_err*n_peaks

    a_guess= -200
    b_guess= 199000000000
    c_guess= -15600000000

    plt.errorbar(peak_times, peaks_freq, yerr=peaks_freq_err, fmt='x', label = 'calibration')
    plt.plot(peak_times, calibration_fit(peak_times, a_guess,b_guess,c_guess), 'r-', label = 'initial guess')
    plt.xlabel('mid peak time (s)')
    plt.ylabel('distance between peaks (s)')
    plt.legend()
    plt.title('Calibration spectrum of fabry perot')
    plt.show()



    #we expect the scanning to not be linear but quadratic, fit the difference in peak (which should be equal) to a line and see the slope. This will be the conversion factor from time to wavelength
    popt_cal, pcov_cal = curve_fit(calibration_fit, peak_times, peaks_freq, p0 = [a_guess,b_guess,c_guess], sigma=peaks_freq_err,maxfev=100000)
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

    return a_cal, b_cal, c_cal, a_cal_err, b_cal_err, c_cal_err
