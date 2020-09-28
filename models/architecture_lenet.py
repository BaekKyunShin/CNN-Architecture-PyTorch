import torch
from torch import nn, Tensor
from torchsummary import summary


class LeNet5(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    model = LeNet5(num_classes=10).to('cpu')
    print(summary(model, input_data=(3, 32, 32), verbose=0))