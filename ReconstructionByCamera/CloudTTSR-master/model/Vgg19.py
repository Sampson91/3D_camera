import torch.nn as torch_neural_network


class Vgg19(torch_neural_network.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        self.block1 = torch_neural_network.Sequential(
            torch_neural_network.Conv1d(in_channels=6, out_channels=64, kernel_size=3,
                      stride=1,
                      padding=1),
            torch_neural_network.BatchNorm1d(num_features=64),
            torch_neural_network.ReLU(),
            torch_neural_network.Conv1d(in_channels=64, out_channels=64, kernel_size=3,
                      stride=1, padding=1),
            torch_neural_network.BatchNorm1d(num_features=64),
            torch_neural_network.ReLU(),
            torch_neural_network.MaxPool1d(kernel_size=2, stride=2)
        )
        self.block2 = torch_neural_network.Sequential(
            torch_neural_network.Conv1d(in_channels=64, out_channels=128, kernel_size=3,
                      stride=1, padding=1),
            torch_neural_network.BatchNorm1d(num_features=128),
            torch_neural_network.ReLU(),
            torch_neural_network.Conv1d(in_channels=128, out_channels=128, kernel_size=3,
                      stride=1, padding=1),
            torch_neural_network.BatchNorm1d(num_features=128),
            torch_neural_network.ReLU(),
            torch_neural_network.MaxPool1d(kernel_size=2, stride=2)
        )
        self.block3 = torch_neural_network.Sequential(
            torch_neural_network.Conv1d(in_channels=128, out_channels=256, kernel_size=3,
                      stride=1, padding=1),
            torch_neural_network.BatchNorm1d(num_features=256),
            torch_neural_network.ReLU(),
            torch_neural_network.Conv1d(in_channels=256, out_channels=256, kernel_size=3,
                      stride=1, padding=1),
            torch_neural_network.BatchNorm1d(num_features=256),
            torch_neural_network.ReLU(),
            torch_neural_network.Conv1d(in_channels=256, out_channels=256, kernel_size=3,
                      stride=1, padding=1),
            torch_neural_network.BatchNorm1d(num_features=256),
            torch_neural_network.ReLU(),
            torch_neural_network.Conv1d(in_channels=256, out_channels=256, kernel_size=3,
                      stride=1, padding=1),
            torch_neural_network.BatchNorm1d(num_features=256),
            torch_neural_network.ReLU(),
            torch_neural_network.MaxPool1d(kernel_size=2, stride=2)
        )
        self.block4 = torch_neural_network.Sequential(
            torch_neural_network.Conv1d(in_channels=256, out_channels=512, kernel_size=3,
                      stride=1, padding=1),
            torch_neural_network.BatchNorm1d(num_features=512),
            torch_neural_network.ReLU(),
            torch_neural_network.Conv1d(in_channels=512, out_channels=512, kernel_size=3,
                      stride=1, padding=1),
            torch_neural_network.BatchNorm1d(num_features=512),
            torch_neural_network.ReLU(),
            torch_neural_network.Conv1d(in_channels=512, out_channels=512, kernel_size=3,
                      stride=1, padding=1),
            torch_neural_network.BatchNorm1d(num_features=512),
            torch_neural_network.ReLU(),
            torch_neural_network.Conv1d(in_channels=512, out_channels=512, kernel_size=3,
                      stride=1, padding=1),
            torch_neural_network.BatchNorm1d(num_features=512),
            torch_neural_network.ReLU(),
            torch_neural_network.MaxPool1d(kernel_size=2, stride=2)
        )
        self.block5 = torch_neural_network.Sequential(
            torch_neural_network.Conv1d(in_channels=512, out_channels=512, kernel_size=3,
                      stride=1, padding=1),
            torch_neural_network.BatchNorm1d(num_features=512),
            torch_neural_network.ReLU()
        )

    def forward(self, super_or_high_resolution):
        sub_super_or_high_resolution_relu5_1 = self.block1(super_or_high_resolution)
        sub_super_or_high_resolution_relu5_1 = self.block2(sub_super_or_high_resolution_relu5_1)
        sub_super_or_high_resolution_relu5_1 = self.block3(sub_super_or_high_resolution_relu5_1)
        sub_super_or_high_resolution_relu5_1 = self.block4(sub_super_or_high_resolution_relu5_1)
        sub_super_or_high_resolution_relu5_1 = self.block5(sub_super_or_high_resolution_relu5_1)
        return sub_super_or_high_resolution_relu5_1



if __name__ == '__main__':
    vgg19 = Vgg19(requires_grad=False)
