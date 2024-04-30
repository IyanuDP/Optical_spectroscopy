import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from numpy import dot
from numpy.linalg import inv
from numpy.linalg import qr

# things that I imported first

deconv = pd.read_csv('Deconv2.csv', index_col=0)
# how I imported the csv file

print(deconv.columns)