import torch
import torch.nn as torch_neural_network
import torch.nn.functional as torch_neural_network_functional


def conv1x1(in_channels, out_channels, stride=1):
    return torch_neural_network.Conv1d(in_channels, out_channels, kernel_size=1,
                                       stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return torch_neural_network.Conv1d(in_channels, out_channels, kernel_size=3,
                                       stride=stride, padding=1, bias=True)


class ResidualBlock(torch_neural_network.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 residual_scale=1):
        super(ResidualBlock, self).__init__()
        self.residual_scale = residual_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = torch_neural_network.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, number_features):
        number_features_1 = number_features
        out = self.conv1(number_features)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.residual_scale + number_features_1
        return out


class Backbone(torch_neural_network.Module):
    def __init__(self, number_residual_blocks, number_features, residual_scale):
        super(Backbone, self).__init__()
        self.number_residual_blocks = number_residual_blocks
        self.conv_head = conv3x3(6, number_features)

        self.residual_blocks = torch_neural_network.ModuleList()
        for i in range(self.number_residual_blocks):
            self.residual_blocks.append(
                ResidualBlock(in_channels=number_features,
                              out_channels=number_features,
                              residual_scale=residual_scale))

        self.conv_tail = conv3x3(number_features, number_features)

    def forward(self, low_resolution_image_feature):
        # F
        low_resolution_image_feature = torch_neural_network_functional.relu(
            self.conv_head(low_resolution_image_feature))
        low_resolution_image_feature_1 = low_resolution_image_feature
        for i in range(self.number_residual_blocks):
            low_resolution_image_feature = self.residual_blocks[i](
                low_resolution_image_feature)
        low_resolution_image_feature = self.conv_tail(
            low_resolution_image_feature)
        low_resolution_image_feature = low_resolution_image_feature + low_resolution_image_feature_1
        return low_resolution_image_feature


class CrossScaleFeatureIntegration2(torch_neural_network.Module):
    def __init__(self, number_features):
        super(CrossScaleFeatureIntegration2, self).__init__()
        self.conv12 = conv1x1(number_features, number_features)
        self.conv21 = conv3x3(number_features, number_features, 2)

        self.conv_merge1 = conv3x3(number_features * 2, number_features)
        self.conv_merge2 = conv3x3(number_features * 2, number_features)

    def forward(self, stage_1, stage_2):
        stage_1_2 = torch_neural_network_functional.interpolate(stage_1,
                                                                scale_factor=2,
                                                                mode='linear')
        stage_1_2 = torch_neural_network_functional.relu(self.conv12(stage_1_2))
        stage_2_1 = torch_neural_network_functional.relu(self.conv21(stage_2))

        stage_1 = torch_neural_network_functional.relu(
            self.conv_merge1(torch.cat((stage_1, stage_2_1), dim=1)))
        stage_2 = torch_neural_network_functional.relu(
            self.conv_merge2(torch.cat((stage_2, stage_1_2), dim=1)))

        return stage_1, stage_2


class CrossScaleFeatureIntegration3(torch_neural_network.Module):
    def __init__(self, number_features):
        super(CrossScaleFeatureIntegration3, self).__init__()
        self.conv12 = conv1x1(number_features, number_features)
        self.conv13 = conv1x1(number_features, number_features)

        self.conv21 = conv3x3(number_features, number_features, 2)
        self.conv23 = conv1x1(number_features, number_features)

        self.conv31_1 = conv3x3(number_features, number_features, 2)
        self.conv31_2 = conv3x3(number_features, number_features, 2)
        self.conv32 = conv3x3(number_features, number_features, 2)

        self.conv_merge1 = conv3x3(number_features * 3, number_features)
        self.conv_merge2 = conv3x3(number_features * 3, number_features)
        self.conv_merge3 = conv3x3(number_features * 3, number_features)

    def forward(self, stage_1, stage_2, stage_3):
        stage_1_2 = torch_neural_network_functional.interpolate(stage_1,
                                                                scale_factor=2,
                                                                mode='linear')
        stage_1_2 = torch_neural_network_functional.relu(self.conv12(stage_1_2))
        stage_1_3 = torch_neural_network_functional.interpolate(stage_1,
                                                                scale_factor=4,
                                                                mode='linear')
        stage_1_3 = torch_neural_network_functional.relu(self.conv13(stage_1_3))

        stage_2_1 = torch_neural_network_functional.relu(self.conv21(stage_2))
        stage_2_3 = torch_neural_network_functional.interpolate(stage_2,
                                                                scale_factor=2,
                                                                mode='linear')
        stage_2_3 = torch_neural_network_functional.relu(self.conv23(stage_2_3))

        stage_3_1 = torch_neural_network_functional.relu(self.conv31_1(stage_3))
        stage_3_1 = torch_neural_network_functional.relu(
            self.conv31_2(stage_3_1))
        stage_3_2 = torch_neural_network_functional.relu(self.conv32(stage_3))

        stage_1 = torch_neural_network_functional.relu(
            self.conv_merge1(torch.cat((stage_1, stage_2_1, stage_3_1), dim=1)))
        stage_2 = torch_neural_network_functional.relu(
            self.conv_merge2(torch.cat((stage_2, stage_1_2, stage_3_2), dim=1)))
        stage_3 = torch_neural_network_functional.relu(
            self.conv_merge3(torch.cat((stage_3, stage_1_3, stage_2_3), dim=1)))

        return stage_1, stage_2, stage_3


class MergeTail(torch_neural_network.Module):
    def __init__(self, number_features):
        super(MergeTail, self).__init__()
        self.conv13 = conv1x1(number_features, number_features)
        self.conv23 = conv1x1(number_features, number_features)
        self.conv_merge = conv3x3(number_features * 3, number_features)
        self.conv_tail1 = conv3x3(number_features, number_features // 2)
        self.conv_tail2 = conv1x1(number_features // 2, 6)

    def forward(self, stage_1, stage_2, stage_3):
        stage_1_3 = torch_neural_network_functional.interpolate(stage_1,
                                                                scale_factor=4,
                                                                mode='linear')
        stage_1_3 = torch_neural_network_functional.relu(self.conv13(stage_1_3))
        stage_2_3 = torch_neural_network_functional.interpolate(stage_2,
                                                                scale_factor=2,
                                                                mode='linear')
        stage_2_3 = torch_neural_network_functional.relu(self.conv23(stage_2_3))

        Cross_Scale_Feature_Intefration_output = torch_neural_network_functional.relu(
            self.conv_merge(torch.cat((stage_3, stage_1_3, stage_2_3), dim=1)))
        Cross_Scale_Feature_Intefration_output = self.conv_tail1(
            Cross_Scale_Feature_Intefration_output)
        Cross_Scale_Feature_Intefration_output = self.conv_tail2(
            Cross_Scale_Feature_Intefration_output)
        # Cross_Scale_Feature_Intefration_output = torch.clamp(
        #     Cross_Scale_Feature_Intefration_output, -1, 1)

        return Cross_Scale_Feature_Intefration_output


class MainNet(torch_neural_network.Module):
    def __init__(self, number_resblocks, number_features, residual_scale):
        super(MainNet, self).__init__()
        self.number_residual_blocks = number_resblocks  # a list containing number of resblocks of different stages
        self.number_features = number_features

        self.Backbone = Backbone(self.number_residual_blocks[0],
                                 number_features,
                                 residual_scale)

        # stage11
        self.conv11_head = conv3x3(256 + number_features, number_features)
        self.residual_block_11 = torch_neural_network.ModuleList()
        for i in range(self.number_residual_blocks[1]):
            self.residual_block_11.append(
                ResidualBlock(in_channels=number_features,
                              out_channels=number_features,
                              residual_scale=residual_scale))
        self.conv11_tail = conv3x3(number_features, number_features)

        # subpixel 1 -> 2
        self.conv12 = conv3x3(number_features, number_features * 4)
        # self.pixelshuffle_12 = torch_neural_network.PixelShuffle(2)
        self.convT1D_12 = torch_neural_network.ConvTranspose1d(256, 64, kernel_size=2, stride=2, padding=0, output_padding=0, bias=False)
        

        # stage21, 22
        # self.conv21_head = conv3x3(number_features, number_features)
        self.conv22_head = conv3x3(128 + number_features, number_features)

        self.Cross_Scale_Feature_Intefration_12_output = CrossScaleFeatureIntegration2(
            number_features)

        self.residual_block_21 = torch_neural_network.ModuleList()
        self.residual_block_22 = torch_neural_network.ModuleList()
        for i in range(self.number_residual_blocks[2]):
            self.residual_block_21.append(
                ResidualBlock(in_channels=number_features,
                              out_channels=number_features,
                              residual_scale=residual_scale))
            self.residual_block_22.append(
                ResidualBlock(in_channels=number_features,
                              out_channels=number_features,
                              residual_scale=residual_scale))

        self.conv21_tail = conv3x3(number_features, number_features)
        self.conv22_tail = conv3x3(number_features, number_features)

        # subpixel 2 -> 3
        self.conv23 = conv3x3(number_features, number_features * 4)
        # self.pixelshuffle_23 = torch_neural_network.PixelShuffle(2)
        self.convT1D_23 = torch_neural_network.ConvTranspose1d(256, 64, kernel_size=2, stride=2, padding=0, output_padding=0, bias=False)


        # stage31, 32, 33
        # self.conv31_head = conv3x3(number_features, number_features)
        # self.conv32_head = conv3x3(number_features, number_features)
        self.conv33_head = conv3x3(64 + number_features, number_features)

        self.Cross_Scale_Feature_Intefration_123_output = CrossScaleFeatureIntegration3(
            number_features)

        self.residual_block_31 = torch_neural_network.ModuleList()
        self.residual_block_32 = torch_neural_network.ModuleList()
        self.residual_block_33 = torch_neural_network.ModuleList()
        for i in range(self.number_residual_blocks[3]):
            self.residual_block_31.append(
                ResidualBlock(in_channels=number_features,
                              out_channels=number_features,
                              residual_scale=residual_scale))
            self.residual_block_32.append(
                ResidualBlock(in_channels=number_features,
                              out_channels=number_features,
                              residual_scale=residual_scale))
            self.residual_block_33.append(
                ResidualBlock(in_channels=number_features,
                              out_channels=number_features,
                              residual_scale=residual_scale))

        self.conv31_tail = conv3x3(number_features, number_features)
        self.conv32_tail = conv3x3(number_features, number_features)
        self.conv33_tail = conv3x3(number_features, number_features)

        self.merge_tail = MergeTail(number_features)

    def forward(self, stage_output, Soft_attention_map=None,
                Hard_attention_output_lv3=None, Hard_attention_output_lv2=None,
                Hard_attention_output_lv1=None):
        # shallow feature extraction
        stage_output = self.Backbone(stage_output)

        # stage11
        stage_1_1 = stage_output

        # soft-attention
        stage_1_1_res = stage_1_1
        stage_1_1_res = torch.cat((stage_1_1_res, Hard_attention_output_lv3),
                                  dim=1)
        stage_1_1_res = self.conv11_head(
            stage_1_1_res)  # F.relu(self.conv11_head(x11_res))
        stage_1_1_res = stage_1_1_res * Soft_attention_map
        stage_1_1 = stage_1_1 + stage_1_1_res

        stage_1_1_res = stage_1_1

        for i in range(self.number_residual_blocks[1]):
            stage_1_1_res = self.residual_block_11[i](stage_1_1_res)
        stage_1_1_res = self.conv11_tail(stage_1_1_res)
        stage_1_1 = stage_1_1 + stage_1_1_res

        # stage21, 22
        stage_2_1 = stage_1_1
        stage_2_1_res = stage_2_1
        stage_2_2 = self.conv12(stage_1_1)
        # stage_2_2 = torch_neural_network_functional.relu(
        #     self.pixelshuffle_12(stage_2_2))
        
        stage_2_2 = torch_neural_network_functional.relu(self.convT1D_12(stage_2_2))

        # soft-attention
        stage_2_2_res = stage_2_2
        stage_2_2_res = torch.cat((stage_2_2_res, Hard_attention_output_lv2),
                                  dim=1)
        stage_2_2_res = self.conv22_head(
            stage_2_2_res)  # F.relu(self.conv22_head(x22_res))
        stage_2_2_res = stage_2_2_res * torch_neural_network_functional.interpolate(
            Soft_attention_map, scale_factor=2, mode='linear')
        stage_2_2 = stage_2_2 + stage_2_2_res

        stage_2_2_res = stage_2_2

        stage_2_1_res, stage_2_2_res = self.Cross_Scale_Feature_Intefration_12_output(
            stage_2_1_res, stage_2_2_res)

        for i in range(self.number_residual_blocks[2]):
            stage_2_1_res = self.residual_block_21[i](stage_2_1_res)
            stage_2_2_res = self.residual_block_22[i](stage_2_2_res)

        stage_2_1_res = self.conv21_tail(stage_2_1_res)
        stage_2_2_res = self.conv22_tail(stage_2_2_res)
        stage_2_1 = stage_2_1 + stage_2_1_res
        stage_2_2 = stage_2_2 + stage_2_2_res

        # stage31, 32, 33
        stage_3_1 = stage_2_1
        stage_3_1_res = stage_3_1
        stage_3_2 = stage_2_2
        stage_3_2_res = stage_3_2
        stage_3_3 = self.conv23(stage_2_2)
        # stage_3_3 = torch_neural_network_functional.relu(
        #     self.pixelshuffle_23(stage_3_3))

        stage_3_3 = torch_neural_network_functional.relu(self.convT1D_23(stage_3_3))

        # soft-attention
        stage_3_3_res = stage_3_3
        stage_3_3_res = torch.cat((stage_3_3_res, Hard_attention_output_lv1),
                                  dim=1)
        stage_3_3_res = self.conv33_head(
            stage_3_3_res)  # F.relu(self.conv33_head(x33_res))
        stage_3_3_res = stage_3_3_res * torch_neural_network_functional.interpolate(
            Soft_attention_map, scale_factor=4, mode='linear')
        stage_3_3 = stage_3_3 + stage_3_3_res

        stage_3_3_res = stage_3_3

        stage_3_1_res, stage_3_2_res, stage_3_3_res = self.Cross_Scale_Feature_Intefration_123_output(
            stage_3_1_res, stage_3_2_res, stage_3_3_res)

        for i in range(self.number_residual_blocks[3]):
            stage_3_1_res = self.residual_block_31[i](stage_3_1_res)
            stage_3_2_res = self.residual_block_32[i](stage_3_2_res)
            stage_3_3_res = self.residual_block_33[i](stage_3_3_res)

        stage_3_1_res = self.conv31_tail(stage_3_1_res)
        stage_3_2_res = self.conv32_tail(stage_3_2_res)
        stage_3_3_res = self.conv33_tail(stage_3_3_res)
        stage_3_1 = stage_3_1 + stage_3_1_res
        stage_3_2 = stage_3_2 + stage_3_2_res
        stage_3_3 = stage_3_3 + stage_3_3_res

        stage_output = self.merge_tail(stage_3_1, stage_3_2, stage_3_3)

        return stage_output
