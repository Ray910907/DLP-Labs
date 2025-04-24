from torch.utils.data import Dataset as torchData
from glob import glob
from torchvision import transforms
from torchvision.datasets.folder import default_loader as imgloader
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class LoadTrainData(torchData):
    """Training Dataset Loader

    Args:
        root: Dataset Path
        partial: Only train pat of the dataset
    """

    def __init__(self, root, partial=1.0):
        super().__init__()

        self.transform = transforms.Compose([
                transforms.ToTensor(), #Convert to tensor
                transforms.Normalize(mean=[0.4816, 0.4324, 0.3845],std=[0.2602, 0.2518, 0.2537]),# Normalize the pixel values
        ])
        self.folder = sorted([os.path.join(root, file) for file in os.listdir(root)])
        #self.folder = glob(os.path.join(root + '/*.png'))
        self.partial = partial

    def __len__(self):
        return int(len(self.folder) * self.partial)
    
    @property
    def info(self):
        return f"\nNumber of Training Data: {int(len(self.folder) * self.partial)}"
    
    def __getitem__(self, index):
        path = self.folder[index]
        return self.transform(imgloader(path))
    
class LoadTestData(torchData):
    """Training Dataset Loader

    Args:
        root: Dataset Path
        partial: Only train pat of the dataset
    """

    def __init__(self, root, partial=1.0):
        super().__init__()

        self.transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.4868, 0.4341, 0.3844],std=[0.2620, 0.2527, 0.2543]),
                        ])
        self.folder = sorted([os.path.join(root, file) for file in os.listdir(root)])
        self.partial = partial

    def __len__(self):
        return int(len(self.folder) * self.partial)
    
    @property
    def info(self):
        return f"\nNumber of Testing Data: {int(len(self.folder) * self.partial)}"
    
    def __getitem__(self, index):
        path = self.folder[index]
        return self.transform(imgloader(path))
    
class LoadMaskData(torchData):
    """Training Dataset Loader

    Args:
        root: Dataset Path
        partial: Only train pat of the dataset
    """

    def __init__(self, root, partial=1.0):
        super().__init__()

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.folder = sorted([os.path.join(root, file) for file in os.listdir(root)])
        self.partial = partial

    def __len__(self):
        return int(len(self.folder) * self.partial)
    
    @property
    def info(self):
        return f"\nNumber of Mask Data For Inpainting Task: {int(len(self.folder) * self.partial)}"
    
    def __getitem__(self, index):
        path = self.folder[index]
        return self.transform(imgloader(path))
    



def gamma_func(mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.
        
        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        
        def linear_decay(ratio):
            return 1 - ratio

        def cosine_decay(ratio):
            return np.cos(ratio * np.pi / 2)

        def square_decay(ratio):
            return 1 - ratio * ratio

        if mode == "linear":
            return linear_decay
        elif mode == "cosine":
            return cosine_decay
        elif mode == "square":
            return square_decay
        else:
            raise ValueError("mode must be 'linear', 'cosine' ,'square'.")
    
def plot_mask_scheduling_function():
    file_path = os.path.join(os.getcwd(), 'msf.png')  # Correctly create a file path

    iter = 50

    cos = gamma_func(mode = 'cosine')
    line = gamma_func(mode = 'linear')
    square = gamma_func(mode = 'square')
    
    cos_values = []
    line_values = []
    square_values = []
    t_values = []
    for i in range(iter):
        ratio = (i + 1) / iter
        t_values.append(ratio)
        cos_values.append(cos(ratio))
        line_values.append(line(ratio))
        square_values.append(square(ratio))
    
    # Plotting the scheduling function
    plt.plot(t_values, cos_values, label='Cosine', color='b')
    plt.plot(t_values, line_values, label='Linear', color='r')
    plt.plot(t_values, square_values, label='Square', color='g')
    plt.xlabel('t/T')
    plt.ylabel('Mask Ratio')
    plt.title('Mask Scheduling over Iterations')
    plt.legend()
    plt.grid(True)
    plt.savefig(file_path)
    plt.close()

def plot_FID(cos_fid,line_fid,square_fid, iterations):
    """
    Plots the FID score over iterations.

    Args:
    - fids (list of float): A list of FID scores calculated at different iterations.
    - iterations (list of int): A list of iteration numbers.
    """
    file_path = os.path.join(os.getcwd(), 'fid.png')

    if len(cos_fid) != len(iterations) or len(line_fid) != len(iterations) or len(square_fid) != len(iterations):
        raise ValueError("length doesn't match!")

    plt.plot(iterations, cos_fid, label='Cosine', color='b')
    plt.plot(iterations, line_fid, label='Linear', color='r')
    plt.plot(iterations, square_fid, label='Square', color='g')
    plt.xlabel('Iterations')
    plt.ylabel('FID Score')
    plt.title('FID Score over Iterations')
    plt.legend()
    plt.grid(True)
    plt.savefig(file_path)
    plt.close()

plot_mask_scheduling_function()