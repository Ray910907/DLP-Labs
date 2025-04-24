import torch
import torch.nn as nn
import torch.nn.functional as F

#basic residual block
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            ])
        
        self.bn = nn.ModuleList([
            nn.BatchNorm2d(out_channels),
            nn.BatchNorm2d(out_channels)
            ])

        self.relu = nn.ReLU(inplace=True)
        #the shortcut connection is added if input dimension is different from the output
        self.down = None
        if stride != 1 or in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    #based on the input and output diminsion to decided add the output or residual
    def forward(self, x):
        identity = x
        if self.down is not None:
            identity = self.down(x)

        out = self.conv[0](x)
        out = self.bn[0](out)
        out = self.relu(out)
        out = self.conv[1](out)
        out = self.bn[1](out)

        out += identity
        out = self.relu(out)

        return out

class ResNet34Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 64

        #initial layer
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #ResNet blocks (block number in each stage is the same to ResNet34)
        self.layer = nn.ModuleList([
        self._make_layer(64, 3),
        self._make_layer(128, 4, stride=2),
        self._make_layer(256, 6, stride=2),
        self._make_layer(512, 3, stride=2)
        ])

    def _make_layer(self, out_channels, blocks, stride=1):
        #make first layer,then set the input channels number to the current number
        layers = [BasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        #make the other layer
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        #return a sequential container
        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.conv(x)
        x0 = self.bn(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)

        x1 = self.layer[0](x0)
        x2 = self.layer[1](x1)
        x3 = self.layer[2](x2)
        x4 = self.layer[3](x3)

        return x0, x1, x2, x3, x4


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.upconv(x)
        #Use interpolate to adjust the size fo fit the size of skip connection
        x = F.interpolate(x, size=(skip.shape[2], skip.shape[3]), mode="bilinear", align_corners=True)
        x = torch.cat((x, skip), dim=1)
        x = self.conv(x)
        return x



class ResNet34_UNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = ResNet34Encoder()
        
        self.encoder = nn.ModuleList([
        #Encoder (extract the intermediate outputs of ResNet34 as skip connections)
            nn.Sequential(resnet.conv, resnet.bn, resnet.relu),
            resnet.layer[0],
            resnet.layer[1],
            resnet.layer[2],
            resnet.layer[3]
            ])

        #Decoder
        self.decoder = nn.ModuleList([
            DecoderBlock(512, 256, 256),
            DecoderBlock(256, 128, 128),
            DecoderBlock(128, 64, 64),
            DecoderBlock(64, 64, 64),
            nn.Conv2d(64, 1, kernel_size=1)
        ])


    def forward(self, x):
        #Encoder path
        x0 = self.encoder[0](x)
        x1 = self.encoder[1](x0)
        x2 = self.encoder[2](x1)
        x3 = self.encoder[3](x2)
        x4 = self.encoder[4](x3)

        #Decoder path (have ship connection)
        d4 = self.decoder[0](x4, x3)
        d3 = self.decoder[1](d4, x2)
        d2 = self.decoder[2](d3, x1)
        d1 = self.decoder[3](d2, x0)
        out = self.decoder[4](d1)

        return F.interpolate(out, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)


#Test model structure
if __name__ == "__main__":
    model = ResNet34_UNet()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(f"Output shape: {y.shape}")
