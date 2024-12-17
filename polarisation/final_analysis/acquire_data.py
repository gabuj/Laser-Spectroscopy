import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def reduce_noise(t, i):
    #smoothen data using savgol filter
    window_size = 21  # Must be odd and greater than the polynomial order
    poly_order = 2
    clean_i = savgol_filter(i, window_size, poly_order)
    return clean_i

    
    

def get_data(filename, background_filename, calibration_filename):
    data= np.loadtxt(filename, delimiter = ',', skiprows = 1)
    background= np.loadtxt(background_filename, delimiter = ',', skiprows = 1)
    calibration= np.loadtxt(calibration_filename, delimiter = ',', skiprows = 1)

    t= np.array(data[:,0])
    intensities= np.array(data[:,1])  - np.array(background[:,1])
    calibration_t= np.array(calibration[:,0])
    calibration_intensities= np.array(calibration[:,1])

    # plt.plot(t, intensities, label = 'data')
    # # plt.plot(t, background[:,1], label = 'background')
    # plt.show()
    
    intensities_2 = reduce_noise(t, intensities)
    calibration_intensities_2 = reduce_noise(calibration_t, calibration_intensities)
    #again
    intensities_2 = reduce_noise(t, intensities_2)
    calibration_intensities_2 = reduce_noise(calibration_t, calibration_intensities_2)
    
    
    plt.plot(t, intensities, label = 'original data')
    plt.plot(t, intensities_2, label = 'smoothed data')
    plt.legend()
    plt.show()
    return t, intensities_2, calibration_t, calibration_intensities_2