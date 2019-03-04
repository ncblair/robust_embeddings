class BaseModel(object):
	def __init__(self):
		pass

	def predict_model_output(self, input_nd):
		raise NotImplementedError("Not Implemented")

	def train_model(self, input_nd, labels_nk):
		raise NotImplementedError("Not Implemented")
