import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const
from scipy.signal import find_peaks
from acquire_data import get_data
from chop import chop
from calibration import calibrate

#possible data
filenamep = 'polarisation/data/pol_front/WF_PP4.csv'
background_filenamep = 'polarisation/data/pol_front/WFBG_PP4.csv'
calibration_filenamep = 'polarisation/data/pol_front/WFCA_PP4.csv'


filename0= 'polarisation/data/pol_front/WF_P04.csv'
background_filename0= 'polarisation/data/pol_front/WFBG_P04.csv'
calibration_filename0= 'polarisation/data/pol_front/WFCA_P04.csv'


filenamem= 'polarisation/data/pol_front/WF_PM4.csv'
background_filenamem= 'polarisation/data/pol_front/WFBG_PM4.csv'
calibration_filenamem= 'polarisation/data/pol_front/WFCA_PM4.csv'

#load data
t, intensities, calibration_t, calibration_intensities = get_data(filenamep, background_filenamep, calibration_filenamep)

#plot data
plt.plot(t, intensities, label = 'data')
plt.show()


#chop data to part you wanna investigate
xmin=0.079
xcenter= 0.083
xmax=0.087
t, intensities = chop(t, intensities, xmin, xmax)
calibration_t, calibration_intensities = chop(calibration_t, calibration_intensities, xmin, xmax)
# print(len(calibration_t))

#calibrate data
a_cal, b_cal, c_cal, a_cal_err, b_cal_err, c_cal_err = calibrate(calibration_t, calibration_intensities)

