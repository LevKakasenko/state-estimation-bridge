"""
Exports the 2D turbulence snapshots from .mat files into a NumPy array.
"""

import scipy.io
import numpy as np
from tqdm import tqdm
import pickle

x = np.empty((128**2,0))

for idx in tqdm(range(0, 1001)):
    idx_str = str(idx)
    idx_str = (4 - len(idx_str))*'0' + idx_str

    data = scipy.io.loadmat('2DTurbData/snapshots/turb_w_' + idx_str + '.mat')
    
    data = data['w']
    data = data.flatten()
    x = np.concatenate((x,data[:,np.newaxis]),axis=1)
    
with open('2DTurbData/2DTurbData.pkl', 'wb') as file:
    pickle.dump(x, file)
