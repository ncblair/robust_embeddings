import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class EmbedLayer(nn.Module):
	def __init__(self, in_feat, out_feat):
		super(EmbedLayer, self).__init__()
		self.weight = nn.Parameter(torch.randn(in_feat, out_feat)*0.0001, requires_grad=False)
		self.bias = nn.Parameter(torch.randn(out_feat) * 2 * np.pi, requires_grad=False)

	def forward(self, input):
		return torch.cos(input @ self.weight + self.bias) * torch.sqrt(torch.tensor(2.0) / self.bias.shape[0])


# layers is a list of dictionaries containing information
# on what layers should be added to the model
# Ex: layers = [
#				{ name : "conv",
#				  shape : (1, 8, 5) } # Assumes max_pool after
#               { name : "fc",
#				  shape : (8*13*13, 100) }
#				{ name : "fc",
#				  shape : (100, 10) }
#				]
class BaseNetwork(nn.Module):
	def __init__(self, model_name, layers):
		super(BaseNetwork, self).__init__()
		self.convs = nn.ModuleList([])
		self.fcs = nn.ModuleList([])
		self.embs = nn.ModuleList([])
		self.name = model_name
		for l in layers:
			if l["name"] == "fc":
				self.fcs.append(nn.Linear(*l["shape"]))
			elif l["name"] == "conv":
				self.convs.append(nn.Conv2d(*l["shape"]))
			elif l["name"] == "emb":
				self.embs.append(EmbedLayer(*l["shape"]))
			else:
				print("Layer name not supported, ignoring")


	def forward(self, x):
		for c in self.convs:
			x = F.max_pool2d(F.relu(c(x)),2)
		x = x.view(x.shape[0], -1)
		for e in self.embs:
			x = e(x)
		for fc in self.fcs[:-1]:
			x = F.relu(fc(x))
		return self.fcs[-1](x)


	def train_model(self, train_loader, epochs, opt, criterion, save=True):
		print("Training Model")
		for epoch in range(epochs):
			run_loss = 0.0
			for i, data in enumerate(tqdm(train_loader)):
				inputs, labels = data
				opt.zero_grad()

				outputs = self.forward(inputs)

				loss = criterion(outputs, labels)
				loss.backward()

				opt.step()

				run_loss += loss.item()
				if i % train_loader.batch_size == train_loader.batch_size - 1:
					print('[%d, %5d] loss: %.5f' %
						  (epoch + 1, i + 1, run_loss / (train_loader.batch_size)))
					run_loss = 0.0
		if save:
			print("Saving model")
			self.save_model(self.name + ".pt")

	def save_model(self, PATH):
		torch.save(self.state_dict(), PATH)

	def load_model(self, PATH):
		self.load_state_dict(torch.load(PATH))

	# Temporary method for testing on Additive White Gaussian Noise
	def eval_on_noise_AWGN(self, test_loader, sigmas):
		print("Evaluating model on AWGN")
		accs = []
		with torch.no_grad():
			for sd in tqdm(sigmas):
				correct = 0
				total = 0
				for data in test_loader:
					images, labels = data
					noise = torch.randn_like(images) * sd

					noisy_images = images + noise

					outputs = self.forward(noisy_images)
					_, pred = torch.max(outputs.data, 1)
					total += labels.size(0)
					correct += (pred == labels).sum().item()
				accs.append(100 * correct / total)
		plt.plot(sigmas.numpy(), accs)
		plt.show()



if __name__ == "__main__":
	# test_dict = [{"name" : "conv", "shape" : (1, 8, 5)}, {"name" : "fc", "shape" : (8*12*12, 100)},\
	# 			{"name" : "fc", "shape" : (100, 10)}]
	test_dict = [{"name" : "conv", "shape" : (1, 8, 5)}, {"name": "emb", "shape": (8*12*12, 5000)},\
				 {"name": "fc", "shape": (5000, 128)}, {"name": "fc", "shape": (128, 100)},\
				 {"name": "fc", "shape": (100, 10)}]
	net = BaseNetwork("test_model", test_dict)
	# test_im = torch.randn(1,1,28,28)
	# print(net(test_im))

	transform = transforms.Compose(
		[transforms.ToTensor(), transforms.Lambda(lambda x: 255*x)])
	test_data = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
	train_data = datasets.MNIST(root='./data', train=True, download=False, transform=transform)

	train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=2)
	test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False, num_workers=1)

	optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
	criterion = nn.CrossEntropyLoss()

	net.train_model(train_loader, 1, optimizer, criterion)
	# net.eval_on_noise_AWGN(test_loader, torch.arange(0, 255, 5.0))
