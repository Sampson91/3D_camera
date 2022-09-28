import torch
import torch.nn as torch_neural_network
import torch.nn.functional as torch_neural_network_functional
from torchvision import models
from utils import MeanShift


class LearnableTextureExtractor(torch.nn.Module):
    def __init__(self, requires_grad=True, rgb_range=1):
        super(LearnableTextureExtractor, self).__init__()

        ### use vgg19 weights to initialize
        vgg_pretrained_features = models.vgg19(pretrained=True).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()

        for layers in range(2):
            # Sequential(
            # (0): Conv2d(3,64,kernel_size=(3,3),stride=(1,1),padding=(1,1))
            # (1): ReLU(inplace=True)
            #  )
            self.slice1.add_module(str(layers), vgg_pretrained_features[layers])
        for layers in range(2, 7):
            # Sequential(
            # (2): Conv2d(64,64,kernel_size=(3,3),stride=(1,1),padding=(1,1))
            # (3): ReLU(inplace=True)
            # (4): MaxPool2d(kernel_size=2, stride=2, padding=0,dilation=1,ceil_mode=False)
            # (5): Conv2d(64,128,kernel_size=(3,3),stride=(1,1),padding=(1,1))
            # (6): ReLU(inplace=True)
            #  )
            self.slice2.add_module(str(layers), vgg_pretrained_features[layers])
        for layers in range(7, 12):
            # Sequential(
            # (7): Conv2d(128,128,kernel_size=(3,3),stride=(1,1),padding=(1,1))
            # (8): ReLU(inplace=True)
            # (9): MaxPool2d(kernel_size=2, stride=2, padding=0,dilation=1,ceil_mode=False)
            # (10): Conv2d(128,256,kernel_size=(3,3),stride=(1,1),padding=(1,1))
            # (11): ReLU(inplace=True)
            #  )
            self.slice3.add_module(str(layers), vgg_pretrained_features[layers])
        if not requires_grad:
            for param in self.slice1.parameters():
                param.requires_grad = requires_grad
            for param in self.slice2.parameters():
                param.requires_grad = requires_grad
            for param in self.slice3.parameters():
                param.requires_grad = requires_grad

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)

    def forward(self, input_image):
        input_image = self.sub_mean(input_image)
        input_image = self.slice1(input_image)
        input_image_lv1 = input_image
        input_image = self.slice2(input_image)
        input_image_lv2 = input_image
        input_image = self.slice3(input_image)
        input_image_lv3 = input_image
        return input_image_lv1, input_image_lv2, input_image_lv3
