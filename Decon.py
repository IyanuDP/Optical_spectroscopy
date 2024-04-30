# Import Libraries needd for data manipulation and analysis
# Numpy helps for numerical functions
# Matplotlib is used for the plots and other related functions
# Pandas would help to read data files

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from numpy import dot
from numpy.linalg import inv
from numpy.linalg import qr

# Import the data in .csv format
deconv = pd.read_csv('Deconv2.csv', index_col=0) # how I imported the csv file

# Create a function that would help to read the data and initiate plots needed for the data process
def plot_wavelength():
    ax = plt.axes()
   
    ax.set_ylabel('Intensity(CPS)')
    ax.set_xlabel('wavelength (nm)')
    line_color = 'blue'
    ax.plot(deconv.index, deconv['S1c / R1'], label='ex 450 nm')

    plt.title('AuNC cyst emission')
    plt.legend()

    plt.tight_layout()
    return ax

# ax = plot_wavelength()
# ax.plot(deconv)
# plt.show()
# plotting  a simple spectra good for absorption and emission

# Data Normalization before further analysis
deconv = deconv / deconv.max()
# ax = plot_wavelength()
# ax.plot(deconv)
# plt.show()
# Normailization of spectra

# Analysis involving 'Wavelength' as 'A' and 'Intensity' as 'b'
A = np.array(deconv.index).reshape(-1, 1)  # Independent variable matrix
b = np.array(deconv['S1c / R1']).reshape(-1, 1)  # Dependent variable matrix

# Solving for x in a simple linear regression model A*x = b
# Using pseudo-inverse for the case of an overdetermined system
x = np.dot(np.linalg.pinv(A), b)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(deconv.index, deconv['S1c / R1'], label='Original Intensity')
plt.plot(deconv.index, deconv['S1c / R1'], label='Normalized Intensity')
plt.plot(deconv.index, np.dot(A, x), label='Fitted Line', color='red')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (CPS)')
plt.title('Spectral Analysis')
plt.legend()
plt.tight_layout()
plt.show()