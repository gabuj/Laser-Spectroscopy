import numpy as np
import matplotlib.pyplot as plt



def get_data(filename, background_filename, calibration_filename):
    data= np.loadtxt(filename, delimiter = ',', skiprows = 1)
    background= np.loadtxt(background_filename, delimiter = ',', skiprows = 1)
    calibration= np.loadtxt(calibration_filename, delimiter = ',', skiprows = 1)

    t= np.array(data[:,0])
    intensities= np.array(data[:,1])  - np.array(background[:,1])
    calibration_t= np.array(calibration[:,0])
    calibration_intensities= np.array(calibration[:,1])

    # plt.plot(t, intensities, label = 'data')
    # plt.plot(t, background[:,1], label = 'background')
    # plt.show()
    return t, intensities, calibration_t, calibration_intensities