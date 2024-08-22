import sys
import os

# Add the parent directory to the system path to resolve the import issue
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import math
from dataset.mnist_dataset import get_mnist_dataloaders
from model.quantum_model import QModel
from adversaries.FGSM import fgsm_attack
from adversaries.bim2 import bim_attack

# Hyperparameters
batch_size = 256
learning_rate = 0.005
epochs = 200

# Get data loaders
train_loader, test_loader = get_mnist_dataloaders(batch_size)

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QModel().to(device)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Lists to store loss, accuracy, and duration per epoch
epoch_losses = []
epoch_accuracies = []
epoch_times = []

model.train()
# Training loop
for epoch in range(epochs):
    start_time = time.time()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device, dtype=torch.float)  # Ensure labels match output size
        optimizer.zero_grad()
        images = bim_attack(model, loss_fn, images, labels, 0.3, 3)
        output_probs = model(images)[:, 1]  # Use only the probability for the positive class (|1⟩)
        #print(output_probs)
        loss = loss_fn(output_probs, labels)
        total_loss += loss.item()
        predict = torch.round(output_probs)
        correct += (predict == labels).sum().item()
        total += labels.size(0)
        loss.backward()
        optimizer.step()

    end_time = time.time()
    epoch_duration = end_time - start_time

    # Calculate average loss and accuracy for the epoch
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total

    # Store the results for plotting later
    epoch_losses.append(avg_loss)
    epoch_accuracies.append(accuracy)
    epoch_times.append(epoch_duration)

    # Print epoch summary
    print(f"Epoch [{epoch + 1}/{epochs}], "
          f"Loss: {avg_loss:.4f}, "
          f"Accuracy: {accuracy:.4f}, "
          f"Time: {epoch_duration:.2f} seconds")

# Save the trained model weights
torch.save(model.state_dict(), "train/qModel.pth")

# Plotting the training loss and accuracy
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(range(1, epochs + 1), epoch_losses, 'g-')
ax2.plot(range(1, epochs + 1), epoch_accuracies, 'b-')

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color='g')
ax2.set_ylabel('Accuracy', color='b')

plt.show()

# Evaluation on test dataset
model.eval()
total_loss, correct, total = 0, 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device, dtype=torch.float).squeeze()  # Ensure labels match output size
        output_probs = model(images)[:, 1]  # Use only the probability for the positive class (|1⟩)
        loss = loss_fn(output_probs, labels)
        total_loss += loss.item()
        predict = torch.round(output_probs)
        correct += (predict == labels).sum().item()
        total += labels.size(0)

print(f"Test Loss: {total_loss / len(test_loader):.4f}, Accuracy: {correct / total:.4f}")