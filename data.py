import torch
import os
import h5py
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader


path = os.path.dirname(__file__)


def load_pneumonia(batch_size_train=16, batch_size_test=64):
	data_transforms = transforms.Compose([
		    transforms.Resize(224),
		    transforms.CenterCrop(224),
		    transforms.Grayscale(num_output_channels=1),
		    transforms.ToTensor(),
		    transforms.Normalize(mean = 0.4739,
		                 std = 0.237),
	])
	train_dataset = datasets.ImageFolder(str(path) + '/pneumonia/train', transform=data_transforms)
	test_dataset = datasets.ImageFolder(str(path) + '/pneumonia/test', transform=data_transforms)

	train_loader = DataLoader(
	    train_dataset,
	    batch_size=batch_size_train,
	    shuffle=True,
	    pin_memory=True
	)

	test_loader = DataLoader(
	    test_dataset,
	    batch_size=batch_size_test,
	    shuffle=False,
	    pin_memory=True
	)
	return train_loader, test_loader


# this is currently an experimental dataset, which is not fully supported
def load_derma(batch_size_train=16, batch_size_test=64):
	filename = str(path) + "/pcamv1/camelyonpatch_level_2_split_train_y.h5"
	f = h5py.File(filename, 'r')
	targets_raw = np.asarray(f['y'])
	targets = np.moveaxis(targets_raw, -1, 1)
	f.close()

	filename = str(path) + "/pcamv1/camelyonpatch_level_2_split_train_x.h5"
	f = h5py.File(filename, 'r')
	data_raw = np.asarray(f['x'])
	data = np.moveaxis(data_raw, -1, 1)
	f.close()

	filename = str(path) + "/pcamv1/camelyonpatch_level_2_split_test_y.h5"
	f = h5py.File(filename, 'r')
	targets_raw = np.asarray(f['y'])
	targets_test = np.moveaxis(targets_raw, -1, 1)
	f.close()

	filename = str(path) + "/pcamv1/camelyonpatch_level_2_split_test_x.h5"
	f = h5py.File(filename, 'r')
	data_raw = np.asarray(f['x'])
	data_test = np.moveaxis(data_raw, -1, 1)
	f.close()

	tensor_x_train = torch.Tensor(data)
	tensor_y_train = torch.Tensor(targets)
	tensor_x_test = torch.Tensor(data_test)
	tensor_y_test = torch.Tensor(targets_test)

	train_dataset = TensorDataset(tensor_x_train, tensor_y_train)
	test_dataset = TensorDataset(tensor_x_test, tensor_y_test)

	train_loader = DataLoader(
	    train_dataset,
	    batch_size=batch_size_train,
	    shuffle=True,
	    pin_memory=True
	)

	test_loader = DataLoader(
	    test_dataset,
	    batch_size=batch_size_test,
	    shuffle=False,
	    pin_memory=True
	)
	return train_loader, test_loader


def load_cifar_10(batch_size_train=16, batch_size_test=64):
	cifar_10_transform = transforms.Compose([
						transforms.ToTensor(),
						transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	train_dataset = datasets.CIFAR10("cifar_10", train=True, download=True, transform=cifar_10_transform)
	test_dataset = datasets.CIFAR10("cifar_10", train=False, download=True, transform=cifar_10_transform)

	train_loader = DataLoader(
	    train_dataset,
	    batch_size=batch_size_train,
	    shuffle=True,
	    pin_memory=True
	)

	test_loader = DataLoader(
	    test_dataset,
	    batch_size=batch_size_test,
	    shuffle=False,
	    pin_memory=True
	)
	return train_loader, test_loader 