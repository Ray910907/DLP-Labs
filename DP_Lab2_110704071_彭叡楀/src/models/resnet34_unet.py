import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm):
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

#basic residual block
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            )

        self.out_layer = nn.ReLU()
        #the shortcut connection is added if input dimension is different from the output
        self.down = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            )
    #based on the input and output diminsion to decided add the output or residual
    def forward(self, x):
        
        res = self.down(x)

        out = self.conv(x)

        out += res
        out = self.out_layer(out)

        return out
#decode part performing upsampling, concat and convolution
class Decode(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm,concat=True):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels + skip_channels, out_channels, kernel_size=2, stride=2)
        self.concat = concat
        self.conv = DoubleConv(out_channels, out_channels, use_batchnorm)

    def forward(self, x, skip):
        x = torch.cat([x, skip], dim=1) if self.concat else x
        sample = self.upsample(x)
        return self.conv(sample)

class ResNet34_UNet(nn.Module):
    def __init__(self,use_batchnorm):
        super().__init__()

        #encoder
        self.conv =nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64) if use_batchnorm else nn.Identity(),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.conv_x = nn.ModuleList([
        self.layer_making(64, 64, 3, use_batchnorm),
        self.layer_making(64, 128, 4, use_batchnorm, stride=2),
        self.layer_making(128, 256, 6, use_batchnorm, stride=2),
        self.layer_making(256, 512, 3, use_batchnorm, stride=2),
        self.layer_making(512, 256, 3, use_batchnorm, stride=1)
        ])

        #decoder
        self.decoder = nn.ModuleList([
            Decode(256, 512, 32, use_batchnorm),
            Decode(32, 256, 32, use_batchnorm),
            Decode(32, 128, 32, use_batchnorm),
            Decode(32, 64, 32, use_batchnorm),
            Decode(32, 0, 32, use_batchnorm, concat=False),
            nn.Conv2d(32, 1, kernel_size=1)
        ])
    
    def layer_making(self, in_channels, out_channels, blocks, use_batchnorm, stride=1):
        #make first conv_x,then set the input channels number to the current number
        layers = nn.Sequential()
        layers.append(BasicBlock(in_channels, out_channels, use_batchnorm, stride))
        #make the other conv_x
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, use_batchnorm))
        #return a sequential container
        return layers

    def forward(self, x):
        x0 = self.conv(x)              
        #print(f"X0: {x0.shape}")
        x1 = self.conv_x[0](x0)        
        #print(f"X1: {x1.shape}")
        x2 = self.conv_x[1](x1)       
        #print(f"X2: {x2.shape}")
        x3 = self.conv_x[2](x2) 
        #print(f"X3: {x3.shape}")
        x4 = self.conv_x[3](x3)
        #print(f"X4: {x4.shape}")
        x5 = self.conv_x[4](x4)  
        #print(f"X5: {x4.shape}")

        d0 = self.decoder[0](x5, x4)
        d1 = self.decoder[1](d0, x3)   
        d2 = self.decoder[2](d1, x2)   
        d3 = self.decoder[3](d2, x1)  
        d4 = self.decoder[4](d3, None)
        out = self.decoder[5](d4)

        return out



#Test model structure
if __name__ == "__main__":
    use_batchnorm = True
    model = ResNet34_UNet(use_batchnorm)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(f"Output shape: {y.shape}")
