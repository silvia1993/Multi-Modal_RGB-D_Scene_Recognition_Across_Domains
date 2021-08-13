import torch.nn as nn
from .networks import UpsampleBasicBlock, conv_norm_relu, init_weights

class Decoder(nn.Module):

    def __init__(self, cfg, encoder):
        super(Decoder, self).__init__()

        self.cfg = cfg
        self.encoder = encoder

        dims = [32, 64, 128, 256, 512, 1024, 2048]

        norm = nn.InstanceNorm2d

        self.up1 = UpsampleBasicBlock(dims[4], dims[3], kernel_size=1, padding=0, norm=norm)
        self.up2 = UpsampleBasicBlock(dims[3], dims[2], kernel_size=1, padding=0, norm=norm)
        self.up3 = UpsampleBasicBlock(dims[2], dims[1], kernel_size=1, padding=0, norm=norm)
        self.up4 = UpsampleBasicBlock(dims[1], dims[1], kernel_size=3, padding=1, norm=norm)

        self.skip_3 = conv_norm_relu(dims[3], dims[3], kernel_size=1, padding=0, norm=norm)
        self.skip_2 = conv_norm_relu(dims[2], dims[2], kernel_size=1, padding=0, norm=norm)
        self.skip_1 = conv_norm_relu(dims[1], dims[1], kernel_size=1, padding=0, norm=norm)

        self.up_image = nn.Sequential(
            nn.Conv2d(64, 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

        init_weights(self.up1, 'normal')
        init_weights(self.up2, 'normal')
        init_weights(self.up3, 'normal')
        init_weights(self.up4, 'normal')
        init_weights(self.skip_3, 'normal')
        init_weights(self.skip_2, 'normal')
        init_weights(self.skip_1, 'normal')
        init_weights(self.up_image, 'normal')



    def forward(self, input):
        
        skip1 = self.skip_1(self.encoder.out['1'])  
        skip2 = self.skip_2(self.encoder.out['2'])
        skip3 = self.skip_3(self.encoder.out['3'])

        upconv4 = self.up1(input) #input = self.encoder.out['4']
        upconv3 = self.up2(upconv4 + skip3)
        upconv2 = self.up3(upconv3 + skip2)
        upconv1 = self.up4(upconv2 + skip1)

        generated_images = self.up_image(upconv1)

        return generated_images