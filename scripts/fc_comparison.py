from models.base_network import BaseNetwork
from file_handling.load_datasets import load_mnist
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


reg_net_dict = [{"name":"fc", "shape":(784,5000)},{"name":"fc", "shape":(5000,100)},
				{"name": "fc", "shape":(100,100)}, {"name":"fc", "shape":(100,10)}]
reg_net = BaseNetwork("reg_net", reg_net_dict)

emb_net_dict = [{"name":"emb","shape":(784, 5000)}, {"name":"fc","shape":(5000,100)},
				{"name": "fc", "shape": (100, 100)}, {"name": "fc", "shape": (100, 10)}]
emb_net = BaseNetwork("emb_net", emb_net_dict)


# train_loader, test_loader = load_mnist(mode="torch")
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: 255*x)])
test_data = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
train_data = datasets.MNIST(root='./data', train=True, download=False, transform=transform)


train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test_data, batch_size=1000, shuffle=False, num_workers=1)


reg_net_opt = optim.SGD(reg_net.parameters(), lr=0.001, momentum=0.9)
emb_net_opt = optim.SGD(emb_net.parameters(), lr=0.015, momentum=0.9)

criterion = nn.CrossEntropyLoss()

# reg_net.train_model(train_loader, 5, reg_net_opt, criterion)
reg_net.load_model("reg_net.pt")
# emb_net.load_model("emb_net.pt")
emb_net.train_model(train_loader, 15, emb_net_opt, criterion)

sigmas = torch.arange(0, 255, 5.0)

_, reg_accs = reg_net.eval_on_noise_AWGN(test_loader, sigmas)
_, emb_accs = emb_net.eval_on_noise_AWGN(test_loader, sigmas)

plt.plot(sigmas.numpy(), reg_accs)
plt.plot(sigmas.numpy(), emb_accs)
plt.title("Fully Connected Network Comparison")
plt.legend(["Regular","Embedded"])
plt.show()
