from config import *
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

def conv_block(in_ch, out_ch, kernel=3, stride=2, padding=1, normalize=True,
               activation=nn.LeakyReLU(0.2, inplace=True)):
    layers = [nn.ReflectionPad2d(padding)] if padding > 0 else []
    layers += [nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=0)]
    if normalize:
        layers += [nn.BatchNorm2d(out_ch, affine=True) if BATCH_SIZE > 1 else nn.InstanceNorm2d(out_ch, affine=True)]
    if activation:
        layers += [activation]
    return layers


def deconv_block(in_ch, out_ch,
                 kernel=3, stride=1, padding=1,
                 normalize=True,
                 activation=True):
    layers = [nn.Upsample(scale_factor=2, mode='bilinear'),
              nn.ReflectionPad2d(padding),
              nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=0)]
    if normalize:
        layers += [nn.BatchNorm2d(out_ch, affine=True) if BATCH_SIZE > 1 else nn.InstanceNorm2d(out_ch, affine=True)]
    if activation:
        layers += [nn.ReLU(inplace=True)]
    return layers


class Res_block(nn.Module):

    def __init__(self, in_ch):
        super().__init__()
        self.block = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(in_ch, in_ch, kernel_size=3),
                                   nn.BatchNorm2d(in_ch, affine=True) if BATCH_SIZE > 1 else nn.InstanceNorm2d(in_ch,
                                                                                                               affine=True),
                                   nn.ReLU(inplace=True),
                                   #                                    nn.ReflectionPad2d(1),
                                   #                                    nn.Conv2d(in_ch, in_ch, kernel_size=3),
                                   #                                    nn.BatchNorm2d(in_ch, affine=True) if BATCH_SIZE > 1 else nn.InstanceNorm2d(in_ch, affine=True)
                                   )

    def forward(self, x):
        return x + self.block(x)


class Discriminator(nn.Module):

    def __init__(self, in_ch=3, features=[64, 128, 256, 512]):
        super().__init__()
        layers = []
        for out_ch in features:
            layers += conv_block(in_ch, out_ch, kernel=3)
            in_ch = out_ch
        layers += conv_block(features[-1], 1, kernel=4, stride=1, padding=0, normalize=False, activation=nn.Sigmoid())
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x).view(-1, 1)


class Generator(nn.Module):

    def __init__(self):
        super().__init__()
        layers = []
        layers += conv_block(3, 32, kernel=5, stride=1, padding=2, activation=nn.ReLU(inplace=True))
        layers += conv_block(32, 64, activation=nn.ReLU(inplace=True))
        layers += conv_block(64, 128, activation=nn.ReLU(inplace=True))
        layers += conv_block(128, 256, activation=nn.ReLU(inplace=True))
        layers += conv_block(256, 512, activation=nn.ReLU(inplace=True))
        # layers += conv_block(512,512, activation=nn.ReLU(inplace=True))
        # layers += conv_block(1024, 1024, kernel=2, stride=1, padding=0, normalize=False, activation=nn.Sigmoid())
        layers += [Res_block(512)] * 1
        # layers += deconv_block(1024, 1024, activation=nn.ReLU(inplace=True))
        # layers += deconv_block(512, 512, activation=nn.ReLU(inplace=True))
        layers += deconv_block(512, 256, activation=nn.ReLU(inplace=True))
        layers += deconv_block(256, 128, activation=nn.ReLU(inplace=True))
        layers += deconv_block(128, 64, activation=nn.ReLU(inplace=True))
        layers += deconv_block(64, 32, activation=nn.ReLU(inplace=True))
        layers += conv_block(32, 3, kernel=5, stride=1, padding=2, normalize=False, activation=nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def test():
    D = Discriminator()
    G = Generator()
    test = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    print('Disriminator - OK' if D(test).shape == torch.Size(
        [BATCH_SIZE, 1]) else f'Disriminator - failed: {D(test).shape}')
    print('Generator - OK' if G(test).shape == torch.Size(
        [BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE]) else f'Disriminator - failed: {G(test).shape}')


test()