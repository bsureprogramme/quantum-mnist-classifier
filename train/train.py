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
import csv
import numpy as np
from dataset.mnist_dataset import get_mnist_dataloaders
from model.quantum_model import QModel
from adversaries.FGSM import fgsm_attack
from adversaries.bim2 import bim_attack

# Hyperparameters
batch_size = 256
learning_rate = 0.005
epochs = 200
num_of_experiments = 5

# Get data loaders
train_loader, test_loader = get_mnist_dataloaders(batch_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

avg_train_losses = np.zeros(epochs)
avg_train_accuracies = np.zeros(epochs)
avg_test_losses = np.zeros(epochs)
avg_test_accuracies = np.zeros(epochs)

for experiment in range(num_of_experiments):
    # Initialize the model, loss function, and optimizer
    model = QModel().to(device)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Lists to store loss, accuracy, and duration per epoch
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    epoch_times = []

    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        correct = 0
        total = 0

        #model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device, dtype=torch.float)  # Ensure labels match output size
            optimizer.zero_grad()
            #images = bim_attack(model, loss_fn, images, labels, 0.3, 3)
            #images = fgsm_attack(model, images, labels, 0.3)
            model.train()
            output_probs = model(images)[:, 1]  # Use only the probability for the positive class (|1⟩)
            #print(output_probs)
            loss = loss_fn(output_probs, labels)
            total_loss += loss.item()
            predict = torch.round(output_probs)
            correct += (predict == labels).sum().item()
            total += labels.size(0)
            loss.backward()
            optimizer.step()
        # Calculate average loss and accuracy for the epoch
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        # Store the results for plotting later
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        # Print epoch summary
        print(f"Epoch [{epoch + 1}/{epochs}], "
            f"Train loss: {avg_loss:.4f}, "
            f"Train accuracy: {accuracy:.4f}")

        total_loss = 0
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device, dtype=torch.float).squeeze()  # Ensure labels match output size
                output_probs = model(images)[:, 1]  # Use only the probability for the positive class (|1⟩)
                loss = loss_fn(output_probs, labels)
                total_loss += loss.item()
                predict = torch.round(output_probs)
                correct += (predict == labels).sum().item()
                total += labels.size(0)

            avg_loss = total_loss / len(test_loader)
            accuracy = correct / total        

        test_losses.append(avg_loss)
        test_accuracies.append(accuracy)

        end_time = time.time()
        epoch_duration = end_time - start_time
        epoch_times.append(epoch_duration)

        # Print epoch summary
        print(f"Epoch [{epoch + 1}/{epochs}], "
            f"Test loss: {avg_loss:.4f}, "
            f"Test accuracy: {accuracy:.4f}, "
            f"Time: {epoch_duration:.2f} seconds")

    # Save the trained model weights
    torch.save(model.state_dict(), "train/weights/qModel{}_{}layers.pth".format(experiment+1, model.n_layers))

    if experiment == 0:
        filemode = 'w+'
    else:
        filemode = 'a'

    with open('gen_data/clean/train_loss_{}layers.csv'.format(model.n_layers), filemode, newline='') as f:
        write = csv.writer(f)
        write.writerow(train_losses)

    with open('gen_data/clean/train_accuracy_{}layers.csv'.format(model.n_layers), filemode, newline='') as f:
        write = csv.writer(f)
        write.writerow(train_accuracies)

    with open('gen_data/clean/test_loss_{}layers.csv'.format(model.n_layers), filemode, newline='') as f:
        write = csv.writer(f)
        write.writerow(test_losses)

    with open('gen_data/clean/test_accuracy_{}layers.csv'.format(model.n_layers), filemode, newline='') as f:
        write = csv.writer(f)
        write.writerow(test_accuracies)

    avg_train_losses = avg_train_losses + np.array(train_losses)
    avg_train_accuracies = avg_train_accuracies + np.array(train_accuracies)
    avg_test_losses = avg_test_losses + np.array(test_losses)
    avg_test_accuracies = avg_test_accuracies + np.array(test_accuracies)

avg_train_losses /= num_of_experiments
avg_train_accuracies /= num_of_experiments
avg_test_losses /= num_of_experiments
avg_test_accuracies /= num_of_experiments

with open('gen_data/clean/train_loss_{}layers.csv'.format(model.n_layers), 'a', newline='') as f:
    np.savetxt(f, [avg_train_losses], delimiter=',')

with open('gen_data/clean/train_accuracy_{}layers.csv'.format(model.n_layers), 'a', newline='') as f:
    np.savetxt(f, [avg_train_accuracies], delimiter=',')

with open('gen_data/clean/test_loss_{}layers.csv'.format(model.n_layers), 'a', newline='') as f:
    np.savetxt(f, [avg_test_losses], delimiter=',')

with open('gen_data/clean/test_accuracy_{}layers.csv'.format(model.n_layers), 'a', newline='') as f:
    np.savetxt(f, [avg_test_accuracies], delimiter=',')

# Plotting the training loss and accuracy
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
line1 = ax1.plot(range(1, epochs + 1, 10), avg_train_losses[::10], linestyle='-', marker='o', color='#16425B', label='Training loss')
line2 = ax2.plot(range(1, epochs + 1, 10), avg_train_accuracies[::10], linestyle='-', marker='s', color='#75DDDD', label='Training accuracy')

ax1.set_xlabel('Epochs', labelpad=10, fontweight='bold')
ax1.set_ylabel('Loss', color='#16425B', labelpad=10, fontweight='bold')
ax2.set_ylabel('Accuracy', color='#75DDDD', labelpad=10, fontweight='bold')

lines = line1 + line2
labs = [l.get_label() for l in lines]
ax1.legend(lines, labs, loc='center right')
plt.show()

# Plotting the test loss and accuracy
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
line1 = ax1.plot(range(1, epochs + 1, 10), avg_test_losses[::10], linestyle='-', marker='o', color='#16425B', label='Test loss')
line2 = ax2.plot(range(1, epochs + 1, 10), avg_test_accuracies[::10], linestyle='-', marker='s', color='#75DDDD', label='Test accuracy')

ax1.set_xlabel('Epochs', labelpad=10, fontweight='bold')
ax1.set_ylabel('Loss', color='#16425B', labelpad=10, fontweight='bold')
ax2.set_ylabel('Accuracy', color='#75DDDD', labelpad=10, fontweight='bold')

lines = line1 + line2
labs = [l.get_label() for l in lines]
ax1.legend(lines, labs, loc='center right')
plt.show()
