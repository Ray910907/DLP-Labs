import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import UNet2DModel
from matplotlib import pyplot as plt

class DDPM(nn.Module):
    def __init__(self, n_object_class):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_encoder = self.label_encoder = nn.Sequential(
            nn.Linear(n_object_class, 512),
            nn.ReLU(),
            nn.LayerNorm(512)
        ).to(self.device)
        self.net = UNet2DModel(
            sample_size=(64,64),  # the target image resolution
            in_channels=3,  # the number of input channels, 3 for RGB images
            out_channels=3,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(128, 256, 512, 512),  # Roughly matching our basic unet example
            down_block_types=( 
                "DownBlock2D",  
                "DownBlock2D",  
                "DownBlock2D",  # a regular ResNet downsampling block
                "AttnDownBlock2D", # a ResNet downsampling block with spatial self-attention
            ), 
            up_block_types=(
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",   # a regular ResNet upsampling block
                "UpBlock2D",
                "UpBlock2D",
            ),
            class_embed_type="identity"
        ).to(self.device)

    def forward(self, img, t, label):
        label = label.to(self.device)
        return self.net(sample=img, timestep=t, class_labels=self.label_encoder(label)).sample