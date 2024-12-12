import numpy as np

#load data
filenamep = 'polarisation/data/pol_front/WF_PP4.csv'
background_filenamep = 'polarisation/data/pol_front/WFBG_PP4.csv'
calibration_filenamep = 'polarisation/data/pol_front/WFCA_PP4.csv'


filename0= 'polarisation/data/pol_front/WF_P04.csv'
background_filename0= 'polarisation/data/pol_front/WFBG_P04.csv'
calibration_filename0= 'polarisation/data/pol_front/WFCA_P04.csv'


filenamem= 'polarisation/data/pol_front/WF_PM4.csv'
background_filenamem= 'polarisation/data/pol_front/WFBG_PM4.csv'
calibration_filenamem= 'polarisation/data/pol_front/WFCA_PM4.csv'


def get_data(filename, background_filename, calibration_filename):
    data= np.loadtxt(filename0, delimiter = ',', skiprows = 1)
    background= np.loadtxt(background_filename, delimiter = ',', skiprows = 1)
    calibration= np.loadtxt(calibration_filename, delimiter = ',', skiprows = 1)

    t= np.array(data[:,0])
    intensities= np.array(data[:,1])  - np.array(background[:,1])
    calibration_t= np.array(calibration[:,0])
    calibration_intensities= np.array(calibration[:,1])
    return t, intensities, calibration_t, calibration_intensities