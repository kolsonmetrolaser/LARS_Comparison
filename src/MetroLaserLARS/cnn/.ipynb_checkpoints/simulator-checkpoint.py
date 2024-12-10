import numpy as np
import pandas as pd
import scipy.interpolate
from tqdm import tqdm
import random
from scipy.interpolate import interp1d

def read_data(path, resolution):
    data = np.load(path)
    x, y = data[0], data[1]
    f = interp1d(x, y) # Obtain interpolation function (input original x (time) and y (signal))
    x_new = np.linspace(x.min(), x.max(), resolution) # Create new x (same x_min and x_max but different number of data points)
    y_new = f(x_new) # Obtain new y (based on new x)

    return x_new, y_new, data[2] # return both new x and new y


class Simulator:
    def __init__(self, path, filename, resolution, maximum):
        self.path = path
        self.filename = filename
        self.max = maximum
        self.resolution = resolution

    def sample_batch(self, indices, verbose=0):
        if verbose:
            indices = tqdm(indices)
        for i in indices:
            x, y, peaks = read_data(f'{self.path}/{self.filename}{random.randint(0,self.max)}.npy', resolution=self.resolution)
            peaks_ratio = np.array(np.where(peaks==1)) / len(peaks)
            yield {'chromatogram': y, 'loc': peaks_ratio[0]}
