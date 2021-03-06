import tensorflow as tf
import numpy as np
import torch
import torch.utils.data as utils
from torch.utils.data import DataLoader, TensorDataset
import pickle

def load_mnist(mode="numpy"):

	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

	if mode == "numpy":
		X_train = x_train.reshape((x_train.shape[0], -1))
		X_test = x_test.reshape((x_test.shape[0], -1))
		Y_train = np.eye(10)[y_train]
		Y_test = np.eye(10)[y_test]
		return (X_train, Y_train), (X_test, Y_test)
	elif mode == "torch":
		X_train = np.expand_dims(x_train, 1)
		X_test = np.expand_dims(x_test, 1)
		y_train = y_train
		y_test = y_test
		# X_test = np.expand_dims(X_test, 1)
		# X_test = np.expand_dims(X_test, 1)
		# print(X_train.shape)
		tensor_x_train = torch.stack([torch.Tensor(x) for x in X_train])
		tensor_y_train = torch.Tensor(y_train).long()

		tensor_x_test = torch.stack([torch.Tensor(x) for x in X_test])
		tensor_y_test = torch.Tensor(y_test).long()

		train_loader = TensorDataset(tensor_x_train,tensor_y_train)
		test_loader = TensorDataset(tensor_x_test,tensor_y_test)

		train_loader = DataLoader(dataset=train, batch_size=64, shuffle=True, num_workers=2)
		test_loader = DataLoader(dataset=test, batch_size=64, shuffle=False, num_workers=1)

		return train_loader, test_loader


def load_cifar(mode="torch"):
	batches = [1, 2, 3, 4, 5]
	X, Y = [], []
	for b in batches:
		with open(f"data/CIFAR-10-C/cifar-10-batches-py/data_batch_{b}", "rb") as f:
			batch = pickle.load(f, encoding="bytes")
		X.append(batch[b"data"])
		Y.append(batch[b"labels"])
	X_train = np.concatenate(X)
	X_train = X_train.reshape((X_train.shape[0], 3, 32, 32))
	Y_train = np.concatenate(Y)

	with open("data/CIFAR-10-C/cifar-10-batches-py/test_batch", "rb") as f:
		test_batch = pickle.load(f, encoding="bytes")
	X_test = test_batch[b"data"]
	Y_test = test_batch[b"labels"]

	tensor_x_train = torch.stack([torch.Tensor(x) for x in X_train])
	tensor_y_train = torch.Tensor(Y_train).long()
	tensor_x_test = torch.stack([torch.Tensor(x) for x in X_test])
	tensor_y_train = torch.Tensor(Y_test).long()
	train = TensorDataset(tensor_x_train, tensor_y_train)
	test = TensorDataset(tensor_x_test, tensor_y_test)
	train_loader = DataLoader(dataset=train, batch_size=64, shuffle=True, num_workers=2)
	test_loader = DataLoader(dataset=test, batch_size=64, shuffle=False, num_workers=1)

	return train_loader, test_loader


def load_tiny(path):
	with open("path", "rb") as f:
		tiny_data = pickle.load(f)
	tiny_data["train"]["data"] = torch.stack([torch.Tensor(i) for i in tiny_data['train']['data']])
	tiny_data["train"]["target"] = torch.Tensor(tiny_data['train']['target']).long()

	tiny_data["test"]["data"] = torch.stack([torch.Tensor(i) for i in tiny_data['test']['data']])
	tiny_data["test"]["target"] = torch.Tensor(tiny_data['test']['target']).long()

	tiny_data["train"] = torch.utils.data.TensorDataset(tiny_data["train"]["data"], tiny_data["train"]["target"])
	tiny_data["test"] = torch.utils.data.TensorDataset(tiny_data["test"]["data"], tiny_data["test"]["target"])

	tiny_data["train"] = torch.utils.data.DataLoader(tiny_data["train"], batch_size=16, shuffle=True, num_workers=2)
	tiny_data["test"] = torch.utils.data.DataLoader(tiny_data["test"], batch_size=1000, shuffle=False, num_workers=2)

	return tiny_data["train"], tiny_data["test"]
