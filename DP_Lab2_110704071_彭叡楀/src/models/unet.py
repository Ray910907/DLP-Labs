import torch
import torch.nn as nn
import torch.nn.functional as F
#double convolution
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        #double_convolution for down sampling
        self.down = nn.ModuleList([
            DoubleConv(in_channels, 64),
            DoubleConv(64, 128),
            DoubleConv(128, 256),
            DoubleConv(256, 512),
            DoubleConv(512, 1024)
        ])
        #maxpool
        self.pool = nn.ModuleList([
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        #up sampling
        self.up = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        ])
        #double convolution
        self.conv = nn.ModuleList([
            DoubleConv(1024, 512),
            DoubleConv(512, 256),
            DoubleConv(256, 128),
            DoubleConv(128, 64)
        ])

        #final 1x1 convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        #contracting Path
        x1 = self.down[0](x)
        x2 = self.down[1](self.pool[0](x1))
        x3 = self.down[2](self.pool[1](x2))
        x4 = self.down[3](self.pool[2](x3))
        x5 = self.down[4](self.pool[3](x4))
        
        #expansive Path
        x = self.up[0](x5)
        x = self.conv[0](torch.cat([x, x4], dim=1))
        x = self.up[1](x)
        x = self.conv[1](torch.cat([x, x3], dim=1))
        x = self.up[2](x)
        x = self.conv[2](torch.cat([x, x2], dim=1))
        x = self.up[3](x)
        x = self.conv[3](torch.cat([x, x1], dim=1))
        #reduce the final feature map to the desired number of output
        x = self.final_conv(x)
        
        return x

#Test model structure
if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=1)
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    print(f"Output shape: {output.shape}")
