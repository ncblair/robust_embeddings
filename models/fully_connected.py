import torch
import numpy as np
from models.base import BaseModel

class FullyConnected(BaseModel):
	def __init__(self, d, k):

		# d is the dimension of the input
		# k is the dimension of the output, number of classes
		self.W = np.random.randn(d + 1, k)

	def predict_model_output(self, X_nd):
		return  self._append_1_vec(X_nd) @ self.W

	def train_model(self, X_nd, Y_nk, reg):
		X_nD = self._append_1_vec(X_nd) # D = d + 1
		self.W = np.linalg.solve(X_nD.T@X_nD + reg * np.eye(X_nD.shape[1]), X_nD.T@Y_nk)
		Y_hat_nk = self.predict_model_output(X_nd)
		return self._accuracy(Y_nk, Y_hat_nk)

	def test_model(self, X_nd, Y_nk):
		Y_hat_nk = self.predict_model_output(X_nd)
		return self._accuracy(Y_nk, Y_hat_nk)

	def _accuracy(self, Y_nk, Y_hat_nk):
		return np.mean(np.argmax(Y_hat_nk, axis=1) == np.argmax(Y_nk, axis=1))

	def _append_1_vec(self, M):
		return np.concatenate((M, np.ones((M.shape[0], 1))), axis=1)