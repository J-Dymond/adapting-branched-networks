# Load libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, transform, exposure, img_as_uint, img_as_float
from skimage.io import imsave, imread
import imageio as io
import torch



test_data = pd.read_csv("data/sat-6/X_test_sat6.csv",header=None).values
test_labels = np.array([pd.read_csv("data/sat-6/y_test.csv",skiprows=1,header=None).values[:,1]]).T

print(test_data.shape, test_labels.shape)

new = np.concatenate((test_data, test_labels),axis=1)

print(new.shape)

np.savetxt("data/sat-6/test.csv", new, delimiter=',',fmt='%1.2f')
