from torch.utils.data import Dataset
import torch
import os
import numpy as np


class Data(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,  transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mp3_path = "data/mp3"
        self.npy_path = "data/npy"
        self.npz_path = "data/npz"

        mp3_files = os.listdir(self.mp3_path)
        self.files = [mp3_file.split('.')[0] for mp3_file in mp3_files]



        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        mp3_file = os.path.join(self.mp3_path,
                                self.files[idx]+".mp3")

        npy_file = os.path.join(self.npy_path,
                                self.files[idx]+".mp3.npy")

        npz_file = os.path.join(self.npz_path,
                                self.files[idx]+".npz")

        npy = torch.from_numpy(np.load(npy_file))
        npz = torch.from_numpy(np.load(npz_file)['data'][0:1,:,:].astype('float')/128)

        sample = {'cqt': npy, 'piano_roll': npz}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":
    d = Data()
