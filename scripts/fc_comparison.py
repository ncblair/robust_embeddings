from models.base_network import BaseNetwork
from file_handling.load_datasets import load_mnist
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


reg_net = BaseNetwork("reg_net", [(3072,5000),(5000,100), (100,100),(100,10)],
						["fc", "fc", "fc", "fc"])

emb_net = BaseNetwork("emb_net", [(3072,5000),(5000,100), (100,100),(100,10)],
					["emb", "fc", "fc", "fc"])


transform = transforms.Compose([transforms.ToTensor()])
test_data = datasets.CIFAR10(root='../cifar-10', train=False, download=False, transform=transform)
train_data = datasets.CIFAR10(root='../cifar-10', train=True, download=False, transform=transform)


train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test_data, batch_size=1000, shuffle=False, num_workers=1)


reg_net_opt = optim.SGD(reg_net.parameters(), lr=0.01, momentum=0.9)
emb_net_opt = optim.SGD(emb_net.parameters(), lr=0.015, momentum=0.9)

criterion = nn.CrossEntropyLoss()

reg_net.train_model(train_loader, 5, reg_net_opt, criterion)

emb_net.train_model(train_loader, 15, emb_net_opt, criterion)

sigmas = torch.arange(0, 255, 5.0)

_, reg_accs = reg_net.eval_on_noise_AWGN(test_loader, sigmas)
_, emb_accs = emb_net.eval_on_noise_AWGN(test_loader, sigmas)

plt.plot(sigmas.numpy(), reg_accs)
plt.plot(sigmas.numpy(), emb_accs)
plt.title("Fully Connected Network Comparison")
plt.legend(["Regular","Embedded"])
plt.show()
