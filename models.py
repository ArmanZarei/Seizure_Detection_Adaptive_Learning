import torch.nn as nn


class LogisticRegressionNet(nn.Module):
  def __init__(self, input_channels):
    super(LogisticRegressionNet, self).__init__()

    self.net = nn.Sequential(
        nn.Linear(input_channels, 2),
    )
  
  def forward(self, x):
    return self.net(x)


class CustomNet(nn.Module):
  def __init__(self, input_channels):
    super(CustomNet, self).__init__()

    self.net = nn.Sequential(
        nn.Linear(input_channels, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(p=0.5),
    )
    self.last_layer = nn.Linear(64, 2)
  
  def forward(self, x):
    return self.last_layer(self.net(x))
