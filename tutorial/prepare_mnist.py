"""Download MNIST and save as safetensors for both Python and Nostos training."""
import torch
from torchvision import datasets, transforms
from safetensors.torch import save_file
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

print("Downloading MNIST...")
train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transforms.ToTensor())

# Images: [N, 28, 28] float32, normalized to [0, 1]
# Labels: [N] int32
train_images = train_dataset.data.float() / 255.0
train_labels = train_dataset.targets.to(torch.int32)
test_images = test_dataset.data.float() / 255.0
test_labels = test_dataset.targets.to(torch.int32)

print(f"Train: {train_images.shape} images, {train_labels.shape} labels")
print(f"Test:  {test_images.shape} images, {test_labels.shape} labels")

save_file({
    "train_images": train_images,
    "train_labels": train_labels,
    "test_images": test_images,
    "test_labels": test_labels,
}, os.path.join(DATA_DIR, "mnist.safetensors"))

print(f"Saved to {DATA_DIR}/mnist.safetensors")
