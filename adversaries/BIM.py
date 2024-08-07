# Code adapated from https://github.com/Harry24k/AEPW-pytorch/blob/master/Adversarial%20examples%20in%20the%20physical%20world.ipynb

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

batch_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QModel()
model.load_state_dict(torch.load('train/qModel.pth', weights_only=True))

loss = nn.BCELoss()
alpha = 1
eps = 0.001
scale = 1

train_loader, test_loader = get_mnist_dataloaders(batch_size)

# Also known as I-FGSM Attack
def basic_iterative_attack(model, loss, images, labels, scale, eps, alpha, iters=0) :
    images = images.to(device)
    labels = labels.to(device)
    clamp_max = 255
    
    if iters == 0 :
        # The paper said min(eps + 4, 1.25*eps) is used as iterations
        iters = int(min(eps + 4, 1.25*eps))
                
    if scale :
        eps = eps / 255
        clamp_max = clamp_max / 255
        
    for i in range(iters) :    
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, Tensor.float(labels)).to(device)
        cost.backward()

        attack_images = images + alpha*images.grad.sign()
        
        # Clip attack images(X')
        # min{255, X+eps, max{0, X-eps, X'}}
        # = min{255, min{X+eps, max{max{0, X-eps}, X'}}}
        
        # a = max{0, X-eps}
        a = torch.clamp(images - eps, min=0)
        # b = max{a, X'}
        b = (attack_images>=a).float()*attack_images + (a>attack_images).float()*a
        # c = min{X+eps, b}
        c = (b > images+eps).float()*(images+eps) + (images+eps >= b).float()*b
        # d = min{255, c}
        images = torch.clamp(c, max=clamp_max).detach_()
            
    return images

print("Attack Image & Predicted Label")

model.eval()

correct = 0
total = 0
originals = []
adversarials = []
predictions = []
labels_list = []

for images, labels in train_loader:
    originals.append(images.cpu().data)
    images = basic_iterative_attack(model, loss, images, labels, scale, eps, alpha).to(device)
    labels_list.append(labels)
    labels = labels.to(device)
    outputs = model(images)
    adversarials.append(images.cpu().data)
    pre = torch.round(outputs)
    predictions.append(pre)
    total += 1
    correct += (pre == labels).sum()
    
cols, rows = 4, 2
figure, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(4,8))
for i in range(cols):
    figure.add_subplot(rows, cols, i+1)
    index = random.randint(0, len(adversarials)-1)
    plt.axis('off')
    plt.title("True: {}/Pred: {}".format(labels_list[index].item(), int(predictions[index].item())))
    plt.imshow(adversarials[index].squeeze(), cmap='gray')
    figure.add_subplot(rows, cols, cols+i+1)
    plt.imshow(originals[index].squeeze(), cmap='gray')
    plt.axis('off')

rowtitles = ['Adversarial Image', 'Original Image']

for ax in axes.ravel():
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)

for ax, row in zip(axes[:,0], rowtitles):
    ax.set_ylabel(row, size='x-large', fontweight='bold')

plt.show()
    
print('Accuracy of test text: %f %%' % (100 * float(correct) / total))