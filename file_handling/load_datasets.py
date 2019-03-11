import tensorflow as tf
import numpy as np
import torch
import torch.utils.data as utils
from torch.utils.data import DataLoader, TensorDataset

def load_mnist(mode="numpy"):

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    X_train = x_train.reshape((x_train.shape[0], -1))
    Y_train = np.eye(10)[y_train]

    X_test = x_test.reshape((x_test.shape[0], -1))
    Y_test = np.eye(10)[y_test]

    if mode == "numpy":
        return (X_train, Y_train), (X_test, Y_test)
    elif mode == "torch":
        tensor_x_train = torch.stack([torch.Tensor(x) for x in X_train])
        tensor_y_train = torch.stack([torch.Tensor(y) for y in Y_train]).long()

        tensor_x_test = torch.stack([torch.Tensor(x) for x in X_test])
        tensor_y_test = torch.stack([torch.Tensor(y) for y in Y_test]).long()

        train = TensorDataset(tensor_x_train,tensor_y_train)
        test = TensorDataset(tensor_x_test,tensor_y_test)

        train_loader = DataLoader(dataset=train, batch_size=64, shuffle=True, num_workers=2)
        test_loader = DataLoader(dataset=test, batch_size=64, shuffle=False, num_workers=1)

        return train_loader, test_loader
