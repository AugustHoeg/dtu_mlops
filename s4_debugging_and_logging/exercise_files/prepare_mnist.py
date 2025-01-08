import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image

# Data loading
mnist_transform = transforms.Compose([transforms.ToTensor()])

dataset_path = "datasets"
train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
test_dataset = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)

train_images = train_dataset.data.unsqueeze(1).float()
test_images = test_dataset.data.unsqueeze(1).float()
train_targets = train_dataset.targets.long()
test_targets = test_dataset.targets.long()

# Normalize images
train_images = train_images / 255.0
test_images = test_images / 255.0

torch.save(train_images, "datasets/MNIST/processed/train_images.pt")
torch.save(train_targets, "datasets/MNIST/processed/train_targets.pt")
torch.save(test_images, "datasets/MNIST/processed/test_images.pt")
torch.save(test_targets, "datasets/MNIST/processed/test_targets.pt")



