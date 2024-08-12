import torch
import torch.nn as nn

torch.manual_seed(42)
class LeNet(nn.Module):
  def __init__(self,num_classes=10):
    super(LeNet, self).__init__()
    self.flatten = nn.Flatten()
    self.features = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size = 5, padding = 2),
        nn.BatchNorm2d(6),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size = 2, stride=2),

        nn.Conv2d(6, 16, kernel_size = 5),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size = 2),
    )
    self.FLC = nn.Sequential(
        nn.Linear(16*5*5, 120),
        nn.Dropout(0.5), # Dropout
        nn.BatchNorm1d(120),
        nn.ReLU(inplace = True),
        nn.Linear(120, 40), # Reduce the number of params (120, 84) to (120, 40)
        nn.Dropout(0.5), # Dropout
        nn.ReLU(inplace = True),
        nn.Linear(40, 10), # (84, 10) -> (40, 10)
    )
  def forward(self, x):
    x = self.features(x)
    x = self.flatten(x)
    x = self.FLC(x)
    return x