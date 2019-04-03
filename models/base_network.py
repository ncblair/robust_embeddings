import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt


class EmbedLayer(nn.Module):
	def __init__(self, in_feat, out_feat):
		super(EmbedLayer, self).__init__()
		self.weight = nn.Parameter(torch.randn(in_feat, out_feat)*0.0001, requires_grad=False)
		self.bias = nn.Parameter(torch.randn(out_feat) * 2 * np.pi, requires_grad=False)

	def forward(self, input):
		return torch.cos(input @ self.weight + self.bias) * torch.sqrt(torch.tensor(2.0) / self.bias.shape[0])


class FlattenLayer(nn.Module):
	def __init__(self):
		super(FlattenLayer, self).__init__()

	def forward(self, input):
		return input.view(input.shape[0], -1)


class BaseNetwork1(nn.Sequential):
	def __init__(self, *args):
		super(BaseNetwork1, self).__init__(*args)

	# def train_model(self, train_loader, epochs, opt, criterion, save=True):
	# 	print("Training Model")
	# 	iterations = []
	# 	losses = []
	# 	train_acc = []
	# 	for epoch in range(epochs):
	# 		run_loss = 0.0
	# 		for i, data in enumerate(tqdm(train_loader)):
	# 			inputs, labels = data
	# 			opt.zero_grad()
	#
	# 			outputs = self.forward(inputs)
	#
	# 			loss = criterion(outputs, labels)
	# 			loss.backward()
	#
	# 			opt.step()
	#
	# 			run_loss += loss.item()
	# 			if i % train_loader.batch_size == train_loader.batch_size - 1:
	# 				print('[%d, %5d] loss: %.5f' %
	# 					  (epoch + 1, i + 1, run_loss / (train_loader.batch_size)))
	# 				run_loss = 0.0
	# 				iterations.append(i + epoch * len(train_loader))
	# 				losses.append(run_loss / train_loader.batch_size)
	# 				train_acc.append(running_train_acc / print_every)
	#
	# 		if i % print_every == print_every - 1:  # print every 1000 mini-batches
	#
	#
	#
	# 			test_acc.append(accuracy(model, testloader))
	# 			print(
	# 				f"epoch: {epoch + 1}, iteration: {i}, train acc: {train_acc[-1] :.2f}, test acc: {test_acc[-1] :.2f}, loss: {loss :.2f}")
	# 			running_train_acc = 0.0
	# 			running_loss = 0.0
	# 	# if save:
	# 	# 	print("Saving model")
	# 	# 	self.save_model(self.name + ".pt")

	def _accuracy(self, dataloader):
		total = 0.0
		num_correct = 0.0
		for data in dataloader:
			inputs, labels = data
			outputs = self.forward(inputs)
			outputs = torch.argmax(outputs, 1)
			total += len(outputs)
			num_correct += torch.sum(outputs == labels).numpy()
		accuracy = num_correct / total
		return accuracy

	def train_model(self, epochs, print_every, train_loader, test_loader, optimizer, criterion, scheduler=None, track_test_acc=False):
		train_acc = [.1]
		test_acc = [.1]
		losses = []
		iterations = [0]
		for epoch in range(epochs):  # loop over the dataset multiple times
			running_train_acc = 0
			running_loss = 0
			if scheduler is not None:
				print(f"Learning Rate: {scheduler.get_lr()}")
			for i, data in enumerate(tqdm(train_loader)):
				# get the inputs
				inputs, labels = data
				# one_hot_labels = torch.zeros(labels.shape[0], 10)
				# print(one_hot_labels.shape, labels.view(-1,1).shape)
				# one_hot_labels.scatter_(1, labels.view(-1,1), 1)
				# one_hot_labels = one_hot_labels.float()
				# zero the parameter gradients
				optimizer.zero_grad()

				# forward + backward + optimize
				outputs = self.forward(inputs)
				# print(outputs)
				# break
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()
				running_train_acc += torch.sum(torch.argmax(outputs, 1) == labels).numpy() / len(outputs)
				running_loss += loss

				# print statistics
				if i % print_every == print_every - 1:  # print every 1000 mini-batches
					iterations.append(i + epoch * len(train_loader))
					losses.append(running_loss / print_every)
					train_acc.append(running_train_acc / print_every)
					if track_test_acc:
						test_acc.append(self._accuracy(test_loader))
						print(
							f"epoch: {epoch + 1}, iteration: {i}, train acc: {train_acc[-1] :.2f}, "
							f"test acc: {test_acc[-1] :.2f}, loss: {loss :.2f}")
					else:
						print(
							f"epoch: {epoch + 1}, iteration: {i}, train acc: {train_acc[-1] :.2f}, "
							f"loss: {loss :.6f}")
					running_train_acc = 0.0
					running_loss = 0.0
			if scheduler is not None:
				scheduler.step()

		print('Finished Training')
		return train_acc, test_acc, losses, iterations

	def save_model(self, PATH):
		torch.save(self.state_dict(), PATH)

	def load_model(self, PATH):
		self.load_state_dict(torch.load(PATH))

	def test_model_once(self, test_loader, noise_function, severity=1):
		# print("Evaluating model once")
		with torch.no_grad():
			correct = 0
			total = 0
			for data in test_loader:
				images, labels = data


				if noise_function:
					images = images.permute(0, 2, 3, 1)
					images = torch.tensor([noise_function(i.numpy(), severity) for i in images]).float()
					# images = torch.tensor(noise_function(images.numpy(), severity)).float()
					# print(images.shape)
					images = images.permute(0, 3, 1, 2)
					# print(images.shape)

				outputs = self.forward(images)
				_, pred = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (pred == labels).sum().item()
		return 100*correct / total

	def test_model(self, test_loader, noise_function, graphics=False):
		accs = []
		for severity in range(1,6):
			accs.append(self.test_model_once(test_loader, noise_function, severity))
		if not graphics:
			return accs

		plt.plot(range(1,6), accs)
		plt.xlabel("Severity")
		plt.ylabel("Accuracy")
		plt.show()



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
		# if save:
		# 	print("Saving model")
		# 	self.save_model(self.name + ".pt")

	def save_model(self, PATH):
		torch.save(self.state_dict(), PATH)

	def load_model(self, PATH):
		self.load_state_dict(torch.load(PATH))

	def test_model_once(self, test_loader, noise_function, severity=1):
		# print("Evaluating model once")
		with torch.no_grad():
			correct = 0
			total = 0
			for data in test_loader:
				images, labels = data


				if noise_function:
					images = images.permute(0, 2, 3, 1)
					images = torch.tensor([noise_function(i.numpy(), severity) for i in images]).float()
					# images = torch.tensor(noise_function(images.numpy(), severity)).float()
					# print(images.shape)
					images = images.permute(0, 3, 1, 2)
					# print(images.shape)

				outputs = self.forward(images)
				_, pred = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (pred == labels).sum().item()
		return 100*correct / total

	def test_model(self, test_loader, noise_function, graphics=False):
		accs = []
		for severity in range(1,6):
			accs.append(self.test_model_once(test_loader, noise_function, severity))
		if not graphics:
			return accs

		plt.plot(range(1,6), accs)
		plt.xlabel("Severity")
		plt.ylabel("Accuracy")
		plt.show()
