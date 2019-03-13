import torch
import torch.nn as nn
import torch.nn.functional as F
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

class BaseNetwork(nn.Module):
	def __init__(self, model_name, shapes, layer_types):
		super(BaseNetwork, self).__init__()
		self.convs = nn.ModuleList([])
		self.fcs = nn.ModuleList([])
		self.embs = nn.ModuleList([])
		self.name = model_name

		assert len(shapes) == len(layer_types)

		for type_, shape in zip(layer_types, shapes):
			if type_ == "fc":
				self.fcs.append(nn.Linear(*shape))
			elif type_ == "conv":
				self.convs.append(nn.Conv2d(*shape))
			elif type_ == "emb":
				self.embs.append(EmbedLayer(*shape))
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
	def eval_on_noise_AWGN(self, test_loader, sigmas, graphics=False):
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
		if not graphics:
			return sigmas.numpy(), accs

		plt.plot(sigmas.numpy(), accs)
		plt.show()
