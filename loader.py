import os
import numpy as np
import torch
from preprosesser import pad_along_axis

def get_data():

    npz_folder = "data/npz"
    npy_folder = "data/npy"

    npy_files = os.listdir(npy_folder)
    npy_files = sorted(npy_files)
    npy_files = [os.path.join(npy_folder, file) for file in npy_files]

    npz_files = os.listdir(npz_folder)
    npz_files = sorted(npz_files)
    npz_files = [os.path.join(npz_folder, file) for file in npz_files]

    # calculate space and allocate memory

    file_number = len(npz_files)

    x_shape = np.load(npy_files[0]).shape
    y_shape = np.load(npz_files[0])['data'].shape

    x_shape = (file_number,1)+x_shape
    y_shape = (file_number,)+y_shape

    x = np.zeros(x_shape)
    y = np.zeros(y_shape)
    y = y[:,0:1,:,:]

    for i in range(file_number):
        # print(i)
        try:
            x[i,0,:,:]=np.load(npy_files[i])
            y[i,0,:,:]=np.load(npz_files[i])['data'][0,:,:]
        except Exception as err:
            print(err)

    y = pad_along_axis(y,2640,axis=3)

    print(x.shape, y.shape)
    return x, y

if __name__ == "__main__":
    x, y = get_data()