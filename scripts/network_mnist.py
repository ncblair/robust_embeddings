import torch.optim as optim
from torch.nn import CrossEntropyLoss
import torch

from models.base_network import BaseNetwork
from file_handling.load_datasets import load_mnist

test_dict = [{"name" : "conv", "shape" : (1, 8, 5)}, {"name": "emb", "shape": (8*12*12, 5000)},\
			 {"name": "fc", "shape": (5000, 128)}, {"name": "fc", "shape": (128, 100)},\
			 {"name": "fc", "shape": (100, 10)}]
net = BaseNetwork("test_net", test_dict)

train_loader, test_loader= load_mnist(mode="torch")

optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
criterion = CrossEntropyLoss()

net.train_model(train_loader, 1, optimizer, criterion)
net.eval_on_noise_AWGN(test_loader, torch.arange(0, 255, 5.0))