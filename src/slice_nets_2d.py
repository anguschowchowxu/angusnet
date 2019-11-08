import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


class Micro_Conv(nn.Module):
    def __init__(self, input_channel):
        super(Micro_Conv, self).__init__()
        self.micro_conv = nn.Sequential(
            nn.BatchNorm2d(input_channel),
            nn.ReLU(),
            nn.Conv2d(input_channel, 192, kernel_size=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 48, kernel_size=3, padding=1)
            )

    def forward(self, x):
        out = self.micro_conv(x)
        return out


class Dense2D_Block(nn.Module):
    def __init__(self, input_channel, layers, grow_channels=48):
        super(Dense2D_Block, self).__init__()
        self.layers = layers
        for i in range(self.layers):
            conv = Micro_Conv(input_channel+grow_channels*i)
            setattr(self, 'forw'+str(i+1), conv)

        self.transition = nn.Sequential(
            nn.Conv2d(input_channel+grow_channels*self.layers, input_channel+grow_channels, kernel_size=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )

    def forward(self, x):
        output_x = x
        for i in range(self.layers):
            conv_unit = getattr(self, 'forw'+str(i+1))
            output = conv_unit(output_x)
            output_x = torch.cat((output_x, output), 1)
        block_output = self.transition(output_x)
        return output_x, block_output


class Upsample2D(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Upsample2D, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.post_conv = nn.Sequential(
            nn.BatchNorm2d(input_channel),
            nn.ReLU(),
            nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1)
            )

    def forward(self, x, extra_x=None):
        out = self.upsample(x)
        if extra_x is not None:
            out = torch.cat((out, extra_x), 1)
        out = self.post_conv(out)
        return out


class Dense2D_UNet_v1(nn.Module):
    def __init__(self, down_channels=[96, 144, 192, 240], up_channels=[768, 384, 96, 96, 64]):
        super(Dense2D_UNet_v1, self).__init__()
        self.pre_conv = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(3, down_channels[0], kernel_size=7, stride=2, padding=3)
            )
        self.pre_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block1 = Dense2D_Block(down_channels[0], layers=6, grow_channels=48)
        self.block2 = Dense2D_Block(down_channels[1], layers=12, grow_channels=48)
        self.block3 = Dense2D_Block(down_channels[2], layers=36, grow_channels=48)
        self.block4 = Dense2D_Block(down_channels[3], layers=24, grow_channels=48)

        up1_in = 48*24 + 48*36 + down_channels[3] + down_channels[2]
        self.upsample1 = Upsample2D(up1_in, up_channels[0])
        up2_in = up_channels[0] + 48*12 + down_channels[1]
        self.upsample2 = Upsample2D(up2_in, up_channels[1])
        up3_in = up_channels[1] + 48*6 + down_channels[0]
        self.upsample3 = Upsample2D(up3_in, up_channels[2])
        up4_in = up_channels[2] + down_channels[0]
        self.upsample4 = Upsample2D(up4_in, up_channels[3])
        self.upsample5 = Upsample2D(up_channels[3], up_channels[4])
        self.outlayer1 = nn.Sequential(
            nn.BatchNorm2d(up_channels[4]),
            nn.ReLU(),
            nn.Conv2d(up_channels[4], 3, kernel_size=1), 
            nn.Sigmoid()
            )
        self.outlayer2 = nn.Sequential(
            nn.BatchNorm2d(up_channels[4]),
            nn.ReLU(),
            nn.Conv2d(up_channels[4], 3, kernel_size=1), 
            nn.Sigmoid()
            )

    def forward(self, x):
        pre_conv = self.pre_conv(x)
        pre_out = self.pre_pool(pre_conv)
        db1_out, trans1 = self.block1(pre_out)
        db2_out, trans2 = self.block2(trans1)
        db3_out, trans3 = self.block3(trans2)
        db4_out, trans4 = self.block4(trans3)
        up1_out = self.upsample1(db4_out, db3_out)
        up2_out = self.upsample2(up1_out, db2_out)
        up3_out = self.upsample3(up2_out, db1_out)
        up4_out = self.upsample4(up3_out, pre_conv)
        up5_out = self.upsample5(up4_out)
        final_out1 = self.outlayer1(up5_out)
        final_out2 = self.outlayer2(up5_out)
        return final_out1, final_out2 # (batch_size, 3, y, x)


def get_dense2d_unet_v1(**_):
    model = globals().get('Dense2D_UNet_v1')
    dense2d_unet = model(down_channels=[96, 144, 192, 240, 288], up_channels=[768, 384, 96, 96, 64])
    return dense2d_unet




class Dense2D_UNet_v2(nn.Module):
    def __init__(self, down_channels=[96, 144, 192, 240], up_channels=[768, 384, 96, 96, 64]):
        super(Dense2D_UNet_v2, self).__init__()
        self.pre_conv = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(3, down_channels[0], kernel_size=7, stride=2, padding=3)
            )
        self.pre_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block1 = Dense2D_Block(down_channels[0], layers=6, grow_channels=48)
        self.block2 = Dense2D_Block(down_channels[1], layers=12, grow_channels=48)
        self.block3 = Dense2D_Block(down_channels[2], layers=36, grow_channels=48)
        self.block4 = Dense2D_Block(down_channels[3], layers=24, grow_channels=48)

        up1_in = 48*24 + 48*36 + down_channels[3] + down_channels[2]
        self.upsample1 = Upsample2D(up1_in, up_channels[0])
        up2_in = up_channels[0] + 48*12 + down_channels[1]
        self.upsample2 = Upsample2D(up2_in, up_channels[1])
        up3_in = up_channels[1] + 48*6 + down_channels[0]
        self.upsample3 = Upsample2D(up3_in, up_channels[2])
        up4_in = up_channels[2] + down_channels[0]
        self.upsample4 = Upsample2D(up4_in, up_channels[3])
        self.upsample5 = Upsample2D(up_channels[3], up_channels[4])
        self.outlayer1 = nn.Sequential(
            nn.BatchNorm2d(up_channels[4]),
            nn.ReLU(),
            nn.Conv2d(up_channels[4], 3, kernel_size=1), 
            nn.Sigmoid()
            )
        self.outlayer2 = nn.Sequential(
            nn.BatchNorm2d(up_channels[4]),
            nn.ReLU(),
            nn.Conv2d(up_channels[4], 3, kernel_size=1), 
            nn.Sigmoid()
            )
        self.outlayer3 = nn.Sequential(
            nn.BatchNorm2d(up_channels[4]),
            nn.ReLU(),
            nn.Conv2d(up_channels[4], 3, kernel_size=1), 
            nn.Sigmoid()
            )
        self.outlayer4 = nn.Sequential(
            nn.BatchNorm2d(up_channels[4]),
            nn.ReLU(),
            nn.Conv2d(up_channels[4], 3, kernel_size=1), 
            nn.Sigmoid()
            )

    def forward(self, x):
        pre_conv = self.pre_conv(x)
        pre_out = self.pre_pool(pre_conv)
        db1_out, trans1 = self.block1(pre_out)
        db2_out, trans2 = self.block2(trans1)
        db3_out, trans3 = self.block3(trans2)
        db4_out, trans4 = self.block4(trans3)
        up1_out = self.upsample1(db4_out, db3_out)
        up2_out = self.upsample2(up1_out, db2_out)
        up3_out = self.upsample3(up2_out, db1_out)
        up4_out = self.upsample4(up3_out, pre_conv)
        up5_out = self.upsample5(up4_out)
        final_out1 = self.outlayer1(up5_out)
        final_out2 = self.outlayer2(up5_out)
        final_out3 = self.outlayer3(up5_out)
        final_out4 = self.outlayer4(up5_out)
        return final_out1, final_out2, final_out3, final_out4 # (batch_size, 3, y, x)


def get_dense2d_unet_v2(**_):
    model = globals().get('Dense2D_UNet_v2')
    dense2d_unet = model(down_channels=[96, 144, 192, 240, 288], up_channels=[768, 384, 96, 96, 64])
    return dense2d_unet
