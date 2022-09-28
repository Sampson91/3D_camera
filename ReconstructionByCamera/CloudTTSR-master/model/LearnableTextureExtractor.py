import torch
import torch.nn as torch_neural_network
import torch.nn.functional as torch_neural_network_functional
from torchvision import models
from utils import MeanShift


class LearnableTextureExtractor(torch.nn.Module):
    def __init__(self, requires_grad=True):
        super().__init__()
        self.slice1 = torch_neural_network.Sequential(
            torch_neural_network.Conv1d(in_channels=6, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch_neural_network.ReLU()
        )
        self.slice2 = torch_neural_network.Sequential(
            torch_neural_network.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch_neural_network.ReLU(),
            torch_neural_network.MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            torch_neural_network.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch_neural_network.ReLU()
        )
        self.slice3 = torch_neural_network.Sequential(
            torch_neural_network.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch_neural_network.ReLU(),
            torch_neural_network.MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            torch_neural_network.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch_neural_network.ReLU()
        )

    def forward(self, input_image):
        input_image = self.slice1(input_image)
        input_image_lv1 = input_image
        input_image = self.slice2(input_image)
        input_image_lv2 = input_image
        input_image = self.slice3(input_image)
        input_image_lv3 = input_image
        return input_image_lv1, input_image_lv2, input_image_lv3
