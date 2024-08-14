import torch
import torchvision
import os

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image 

dataset_root = os.getcwd() + '/dataset'

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
dataset = ImageFolder(root=dataset_root, transform=transform)

batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

def visualize_images(images, title):
    fig, axes = plt.subplots(1, len(images), figsize=(12, 4))
    for i, image in enumerate(images):
        axes[i].imshow(np.transpose(image.numpy(), (1, 2, 0)))
        axes[i].axis('off')
    fig.suptitle(title)
    plt.show()

sample_batch, _ = next(iter(data_loader))

visualize_images(sample_batch, title='Original Images')

transformed_images = []
for image in sample_batch:
    pil_image = transforms.ToPILImage()(image)
    transformed_image = transform(pil_image)
    transformed_images.append(transformed_image)

visualize_images(transformed_images, title='Transformed Images')