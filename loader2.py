from torch.utils.data import Dataset
import torch
import os
import numpy as np


class Data(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, transform=None):
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
                                self.files[idx] + ".mp3")

        npy_file = os.path.join(self.npy_path,
                                self.files[idx] + ".mp3.npy")

        npz_file = os.path.join(self.npz_path,
                                self.files[idx] + ".npz")

        npy = np.load(npy_file)
        npz = np.load(npz_file)['data'][:1, :, 2:3].astype('float')

        sample = {'cqt': npy, 'piano_roll': npz}

        if self.transform:
            sample = self.transform(sample)

        return sample


def transform(sample):
    sample['cqt'] = (sample['cqt'] + 100) / 100
    sample['piano_roll'] = sample['piano_roll'] / 50
    return sample


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    d = Data(transform=transform)
    trainloader = torch.utils.data.DataLoader(d, batch_size=5,
                                              shuffle=True, num_workers=4)

    a = [x['piano_roll'].shape for x in trainloader]
    b = [x['cqt'] for x in trainloader]
