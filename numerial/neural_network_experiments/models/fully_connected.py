import torch.nn as nn
import torch.nn.functional as F


class FullyConnectedMNIST(nn.Module):
    def __init__(self, p=0.5):
        super(FullyConnectedMNIST, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


class SimpleFCN(nn.Module):
    def __init__(self):
        super(SimpleFCN, self).__init__()

        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 10)

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity="linear")

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class two_layer_fc(nn.Module):

    def __init__(self):
        super(two_layer_fc, self).__init__()
        self.fc1 = nn.Linear(784, 128)

        self.fc2 = nn.Linear(128, 64)

        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)
        return x
