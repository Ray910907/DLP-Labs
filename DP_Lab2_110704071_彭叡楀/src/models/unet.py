import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

#doing upsampling and concat in a network
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x, concat):
        out = self.up(x)

        return torch.cat([out, concat], dim=1)

class UNet(nn.Module):
    def __init__(self, use_batchnorm, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        self.down = nn.ModuleList([
            DoubleConv(in_channels, 64, use_batchnorm),
            DoubleConv(64, 128, use_batchnorm),
            DoubleConv(128, 256, use_batchnorm),
            DoubleConv(256, 512, use_batchnorm),
            DoubleConv(512, 1024, use_batchnorm)
        ])

        self.pool = nn.ModuleList([
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])

        self.up = nn.ModuleList([
            Upsample(1024, 512),
            Upsample(512, 256),
            Upsample(256, 128),
            Upsample(128, 64),
        ])

        self.conv = nn.ModuleList([
            DoubleConv(1024, 512, use_batchnorm),
            DoubleConv(512, 256, use_batchnorm),
            DoubleConv(256, 128, use_batchnorm),
            DoubleConv(128, 64, use_batchnorm)
        ])


        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):

        x1 = self.down[0](x)
        x2 = self.down[1](self.pool[0](x1))
        x3 = self.down[2](self.pool[1](x2))
        x4 = self.down[3](self.pool[2](x3))
        x5 = self.down[4](self.pool[3](x4))

        x = self.up[0](x5,x4)
        x = self.conv[0](x)
        x = self.up[1](x,x3)
        x = self.conv[1](x)
        x = self.up[2](x,x2)
        x = self.conv[2](x)
        x = self.up[3](x,x1)
        x = self.conv[3](x)

        x = self.final_conv(x)
        
        return x

if __name__ == "__main__":
    use_batchnorm = True
    model = UNet(use_batchnorm, in_channels=3, out_channels=1)
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    print(f"Output shape: {output.shape}")
