# models/__init__.py

from .cnn import new_ResNet18
from .fully_connected import FullyConnectedMNIST, SimpleFCN, two_layer_fc

__all__ = [
    'new_ResNet18',
    'FullyConnectedMNIST',
    'SimpleFCN',
    'two_layer_fc'
]
