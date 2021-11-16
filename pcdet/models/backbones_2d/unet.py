import time
import torch
import torch.nn as nn
import warnings
from torch import cuda
from .meta_kernel import MetaKernel, EdgeConvKernel ,MetaKernelV6

class SALSANEXT(nn.Module):
    """Backbone network for (range/cylinder/bev) projected points.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        meta_kernel_cfg (dict): Meta kernel config.
    """

    def __init__(self,
                 in_channels=5,
                 out_channels=32,
                 append_far=False,
                 kernel_cfg=None):
        super(SALSANEXT, self).__init__()

        self.downCntx = ResContextBlock(in_channels, 32)
        self.downCntx2 = ResContextBlock(32, 32)
        self.downCntx3 = ResContextBlock(32, 32)

        self.use_kernel = (kernel_cfg is not None)
        self.append_far=append_far
        if self.append_far:
            self.downCntx2=ResContextBlock(64,32)
        self.kernel_type=None
        if self.use_kernel:
            self.kernel_type=kernel_cfg.pop("TYPE")
        if self.use_kernel and 'KERNEL_LAYER_INDEX' in kernel_cfg:
            self.kernel_layer_index=kernel_cfg.pop('KERNEL_LAYER_INDEX')
        else:
            self.kernel_layer_index=1

        if self.kernel_type=="meta":
            self.kernel = MetaKernel(kernel_cfg)
        elif self.kernel_type=="edge_conv":
            self.kernel=EdgeConvKernel(kernel_cfg)
        elif self.kernel_type=="meta_v6":
            self.kernel=MetaKernelV6(kernel_cfg)

        self.resBlock1 = ResBlock(
            32, 2 * 32, 0.2, pooling=True, drop_out=False)
        self.resBlock2 = ResBlock(2 * 32, 2 * 2 * 32, 0.2, pooling=True)

        self.resBlock3 = ResBlock(2 * 2 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock4 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock5 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=False)

        self.upBlock1 = UpBlock(2 * 4 * 32, 4 * 32, 4 * 32 * 2, 0.2)
        self.upBlock2 = UpBlock(4 * 32, 4 * 32, 4 * 32 * 2, 0.2)
        self.upBlock3 = UpBlock(4 * 32, 2 * 32, 4 * 32, 0.2)
        self.upBlock4 = UpBlock(2 * 32, out_channels,
                                32 * 2, 0.2, drop_out=False)

    def forward(self, data_dict):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            up1e (torch.Tensor): encodered features with shape (N, C2, H, W).
        """
        
        x = data_dict['points_img']
        mask = data_dict['proj_masks']
        if self.append_far:
            x_far=data_dict['points_img_far']
            mask_far=data_dict['proj_masks_far']

        if self.use_kernel and self.kernel_layer_index == 0:
            mask = mask.unsqueeze(1)
            coord =torch.cat([x[:,:3],x[:,4:5]],dim=1)
            downCntx_coord = torch.cat([coord, x], dim=1)
            x = self.kernel(downCntx_coord, mask)
            if self.append_far:
                mask_far = mask_far.unsqueeze(1)
                coord_far =torch.cat([x_far[:,:3],x_far[:,4:5]],dim=1)
                downCntx_coord_far = torch.cat([coord_far, x_far], dim=1)
                x_far = self.kernel(downCntx_coord_far, mask_far)
                x=torch.cat([x,x_far,],dim=1)

        downCntx = self.downCntx(x)  # N, 32, H, W
        if self.append_far:
            downCntx_far=self.downCntx(x_far)

        if self.use_kernel and self.kernel_layer_index == 1:
            mask=mask.unsqueeze(1)
            coord =torch.cat([x[:,:3],x[:,4:5]],dim=1)
            downCntx_coord = torch.cat([coord, downCntx], dim=1)
            downCntx = self.kernel(downCntx_coord, mask)
            if self.append_far:
                mask_far = mask_far.unsqueeze(1)
                coord_far =torch.cat([x_far[:,:3],x_far[:,4:5]],dim=1)
                downCntx_coord_far = torch.cat([coord_far, downCntx_far], dim=1)
                downCntx_far = self.kernel(downCntx_coord_far, mask_far)
                downCntx=torch.cat([downCntx,downCntx_far],dim=1)

        downCntx = self.downCntx2(downCntx)  # N, 32, H, W
        downCntx = self.downCntx3(downCntx)  # N, 32, H, W

        down0c, down0b = self.resBlock1(downCntx)
        down1c, down1b = self.resBlock2(down0c)

        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down5c = self.resBlock5(down3c)

        up4e = self.upBlock1(down5c, down3b)
        up3e = self.upBlock2(up4e, down2b)
        up2e = self.upBlock3(up3e, down1b)
        up1e = self.upBlock4(up2e, down0b)
        
        if self.use_kernel and self.kernel_layer_index == -1:
            mask=mask.unsqueeze(1)
            coord=x[:, :3, :, :]
            up1e_coord = torch.cat([coord, up1e], dim=1)
            up1e = self.kernel(up1e_coord, mask)
        
        data_dict['fv_features'] = up1e
        return data_dict


class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(ResContextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters,
                               kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3, 3), padding=1)
        self.act2 = nn.LeakyReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters,
                               (3, 3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(out_filters)

    def forward(self, x):

        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(shortcut)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        output = shortcut + resA2
        return output


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3), stride=1,
                 pooling=True, drop_out=True):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(in_filters, out_filters,
                               kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_filters, out_filters,
                               kernel_size=(3, 3), padding=1)
        self.act2 = nn.LeakyReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters,
                               kernel_size=(3, 3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters, out_filters,
                               kernel_size=(2, 2), dilation=2, padding=1)
        self.act4 = nn.LeakyReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv5 = nn.Conv2d(out_filters*3, out_filters, kernel_size=(1, 1))
        self.act5 = nn.LeakyReLU(inplace=True)
        self.bn4 = nn.BatchNorm2d(out_filters)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(
                kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        resA = self.conv4(resA2)
        resA = self.act4(resA)
        resA3 = self.bn3(resA)

        concat = torch.cat((resA1, resA2, resA3), dim=1)
        resA = self.conv5(concat)
        resA = self.act5(resA)
        resA = self.bn4(resA)
        resA = shortcut + resA

        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            resB = self.pool(resB)

            return resB, resA
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB


class AttentionBlock(nn.Module):
    def __init__(self, in_filters_x, in_filters_g, int_filters):
        super(AttentionBlock, self).__init__()
        self.Wx = nn.Sequential(nn.Conv2d(in_filters_x, int_filters, kernel_size=1),
                                nn.BatchNorm2d(int_filters))
        self.Wg = nn.Sequential(nn.Conv2d(in_filters_g, int_filters, kernel_size=1),
                                nn.BatchNorm2d(int_filters))
        self.psi = nn.Sequential(nn.Conv2d(int_filters, 1, kernel_size=1),
                                 nn.BatchNorm2d(1),
                                 nn.Sigmoid())

    def forward(self, x, g):
        # apply the Wx to the skip connection
        x1 = self.Wx(x)
        # after applying Wg to the input, upsample to the size of the skip connection
        g1 = nn.functional.interpolate(
            self.Wg(g), x1.shape[2:], mode='bilinear', align_corners=False)
        out = self.psi(nn.ReLU(inplace=True)(x1 + g1))
        out = nn.Sigmoid()(out)
        return out * x


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, skip_filters, dropout_rate, drop_out=True, with_attention=False):
        super(UpBlock, self).__init__()
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.with_attention = with_attention
        if self.with_attention:
            self.attention = AttentionBlock(
                skip_filters, in_filters, int(skip_filters / 2))

        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.conv1 = nn.Conv2d(in_filters//4 + skip_filters,
                               out_filters, (3, 3), padding=1)

        self.act1 = nn.LeakyReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv2 = nn.Conv2d(out_filters, out_filters,
                               (3, 3), dilation=2, padding=2)
        self.act2 = nn.LeakyReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters,
                               (2, 2), dilation=2, padding=1)
        self.act3 = nn.LeakyReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters*3, out_filters, kernel_size=(1, 1))
        self.act4 = nn.LeakyReLU(inplace=True)
        self.bn4 = nn.BatchNorm2d(out_filters)

        self.dropout3 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x, skip):
        upA = nn.PixelShuffle(2)(x)
        if self.drop_out:
            upA = self.dropout1(upA)

        if skip is not None:
            if self.with_attention:
                skip = self.attention(skip, x)
            upB = torch.cat((upA, skip), dim=1)
            if self.drop_out:
                upB = self.dropout2(upB)
        else:
            upB = upA

        upE = self.conv1(upB)
        upE = self.act1(upE)
        upE1 = self.bn1(upE)

        upE = self.conv2(upE1)
        upE = self.act2(upE)
        upE2 = self.bn2(upE)

        upE = self.conv3(upE2)
        upE = self.act3(upE)
        upE3 = self.bn3(upE)

        concat = torch.cat((upE1, upE2, upE3), dim=1)
        upE = self.conv4(concat)
        upE = self.act4(upE)
        upE = self.bn4(upE)
        if self.drop_out:
            upE = self.dropout3(upE)

        return upE
