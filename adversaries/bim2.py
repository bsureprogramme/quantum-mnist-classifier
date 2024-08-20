import torch
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt
import random
import os
import sys

# Add the parent directory to the system path to resolve the import issue
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from model.quantum_model import QModel
from dataset.mnist_dataset import get_mnist_dataloaders

def bim_attack(model, criterion, data, target, epsilon, iterations):
    data.requires_grad = True
    target = torch.Tensor.float(target)
    perturbed_data = data
    alpha = epsilon/iterations
    
    for i in range(iterations):
        output = model(data)[:, 1]
        model.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        data_grad = data.grad.data
        delta = alpha * data_grad.sign()
        perturbed_data = perturbed_data + delta
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    return perturbed_data

def main():
    model = QModel()
    model.eval()
    _, test_loader = get_mnist_dataloaders()
    for image, label in test_loader:
        image = bim_attack(model, nn.BCELoss(), image, label, 0.3, 3)
    plt.imshow(image[0].detach().squeeze(), cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()