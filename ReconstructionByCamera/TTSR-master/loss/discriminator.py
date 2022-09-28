import torch
import torch.nn as torch_neural_network
import torch.nn.functional as torch_neural_network_functional


def conv3x3(in_channels, out_channels, stride=1):
    return torch_neural_network.Conv2d(in_channels, out_channels, kernel_size=3,
                                       stride=stride, padding=1, bias=True)


class Discriminator(torch_neural_network.Module):
    def __init__(self, in_size=160):
        super(Discriminator, self).__init__()
        self.conv1 = conv3x3(3, 32)
        self.LeakyReLU1 = torch_neural_network.LeakyReLU(0.2)
        self.conv2 = conv3x3(32, 32, 2)
        self.LeakyReLU2 = torch_neural_network.LeakyReLU(0.2)
        self.conv3 = conv3x3(32, 64)
        self.LeakyReLU3 = torch_neural_network.LeakyReLU(0.2)
        self.conv4 = conv3x3(64, 64, 2)
        self.LeakyReLU4 = torch_neural_network.LeakyReLU(0.2)
        self.conv5 = conv3x3(64, 128)
        self.LeakyReLU5 = torch_neural_network.LeakyReLU(0.2)
        self.conv6 = conv3x3(128, 128, 2)
        self.LeakyReLU6 = torch_neural_network.LeakyReLU(0.2)
        self.conv7 = conv3x3(128, 256)
        self.LeakyReLU7 = torch_neural_network.LeakyReLU(0.2)
        self.conv8 = conv3x3(256, 256, 2)
        self.LeakyReLU8 = torch_neural_network.LeakyReLU(0.2)
        self.conv9 = conv3x3(256, 512)
        self.LeakyReLU9 = torch_neural_network.LeakyReLU(0.2)
        self.conv10 = conv3x3(512, 512, 2)
        self.LeakyReLU10 = torch_neural_network.LeakyReLU(0.2)

        self.fully_connected_1 = torch_neural_network.Linear(
            in_size // 32 * in_size // 32 * 512, 1024)
        self.LeakyReLU11 = torch_neural_network.LeakyReLU(0.2)
        self.fully_connected_2 = torch_neural_network.Linear(1024, 1)

    def forward(self, train_crop_size_x4):
        train_crop_size_x4 = self.LeakyReLU1(self.conv1(train_crop_size_x4))
        train_crop_size_x4 = self.LeakyReLU2(self.conv2(train_crop_size_x4))
        train_crop_size_x4 = self.LeakyReLU3(self.conv3(train_crop_size_x4))
        train_crop_size_x4 = self.LeakyReLU4(self.conv4(train_crop_size_x4))
        train_crop_size_x4 = self.LeakyReLU5(self.conv5(train_crop_size_x4))
        train_crop_size_x4 = self.LeakyReLU6(self.conv6(train_crop_size_x4))
        train_crop_size_x4 = self.LeakyReLU7(self.conv7(train_crop_size_x4))
        train_crop_size_x4 = self.LeakyReLU8(self.conv8(train_crop_size_x4))
        train_crop_size_x4 = self.LeakyReLU9(self.conv9(train_crop_size_x4))
        train_crop_size_x4 = self.LeakyReLU10(self.conv10(train_crop_size_x4))

        train_crop_size_x4 = train_crop_size_x4.view(train_crop_size_x4.size(0),
                                                     -1)
        train_crop_size_x4 = self.LeakyReLU11(
            self.fully_connected_1(train_crop_size_x4))
        train_crop_size_x4 = self.fully_connected_2(train_crop_size_x4)

        return train_crop_size_x4


if __name__ == '__main__':
    model = discriminator()
    out_size = torch.ones(1, 3, 160, 160)
    out = model(out_size)
    print(out.size())
