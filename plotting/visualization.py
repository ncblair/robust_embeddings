import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.manifold import TSNE

cifar_classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def visualize_filters(layer, name=None):
	layer1 = layer.weight.data
	supervised_filters = [f - np.min(f.numpy()) for f in layer1]
	supervised_filters = torch.Tensor([f / np.max(f.numpy()) for f in supervised_filters])[:, :3, :, :]
	grid = torchvision.utils.make_grid(supervised_filters).numpy()
	plt.imshow(np.transpose(grid, (1, 2, 0)))
	plt.axis("off")
	if name == None:
		plt.show()
	else:
		plt.savefig(f"output/filters/{name}")

class SaveFeatures():
	features=None
	def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
	def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
	def remove(self): self.hook.remove()

def class_activation_map(model, image, last_convolutional_layer, fc_layer, label, clas=None, name=None):

	# returns an num_inputs x num_classes x h x w array of class activation maps 
	# and a list of predicted labels
	
	if not clas:
		clas = label

	activations = SaveFeatures(last_convolutional_layer)
	output = model(image)[0]
	activations.remove()
	prediction = torch.argmax(output)
	f = activations.features
	wc = fc_layer.weight.data[label].numpy()
	M = np.tensordot(wc, f, axes=1)
	M = M - np.min(M)
	M = M / np.max(M)
	M = resize(M, image.shape[1:], anti_aliasing=True)

	fig, axes = plt.subplots(1, 2)
	axes[0].set_title("original_image")
	axes[0].imshow(image)
	axes[1].set_title(f"Class activation map {clas}")
	axes[1].imshow(image)
	axes[1].imshow(M, alpha=.5, cmap="jet")

	if name:
		plt.savefig(f"output/CAM/{name}")
	else:
		plt.show()

def tsne(images, labels, representation_layer, model, name=None):
	activations = SaveFeatures(representation_layer)
	outputs = model(images)
	activations.remove()
	activations = np.squeeze(activations.features)
	tsne_embedded = TSNE(n_components=2).fit_transform(activations)
	plt.scatter(tsne_embedded[:, 0], tsne_embedded[:, 1], c=labels)
	plt.title("tsne embedding of supervised model learned representation")
	if name:
		plt.savefig(f"output/tsne/{name}")
	else:
		plt.show()

def visualize_activations(images, labels, layer, model, name=None):
	activations = SaveFeatures(layer)
	outputs = model(images)
	activations.remove()
	activations = np.squeeze(activations.features)
	plt.scatter(activations[:, 0], activations[:, 1], c=labels)
	plt.title("tsne embedding of supervised model learned representation")
	if name:
		plt.savefig(f"output/activations/{name}")
	else:
		plt.show()

