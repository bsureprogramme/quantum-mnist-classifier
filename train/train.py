import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from dataset.mnist_dataset import get_mnist_dataloaders
from model.quantum_model import HybridModel

# Hyperparameters
batch_size = 32
n_epochs = 200
learning_rate = 0.001

# Get data loaders
train_loader, test_loader = get_mnist_dataloaders(batch_size)

# Initialize the model, loss function, and optimizer
model = HybridModel().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Lists to store loss and accuracy
train_losses = []
train_accuracies = []

# Training loop
for epoch in range(n_epochs):
    start_time = time.time()
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.view(-1, 16 * 16).to(model.fc1.weight.device)
        labels = labels.to(model.fc1.weight.device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    end_time = time.time()
    epoch_duration = end_time - start_time

    train_losses.append(epoch_loss / len(train_loader))
    train_accuracies.append(100 * correct / total)
    print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {epoch_loss / len(train_loader):.4f}, "
          f"Accuracy: {100 * correct / total:.2f}%, Duration: {epoch_duration:.2f} seconds")

# Save the trained model weights
torch.save(model.state_dict(), "quantum_mnist_model.pth")

# Load the trained model weights for evaluation
model.load_state_dict(torch.load("quantum_mnist_model.pth"))

# Evaluation on test dataset
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.view(-1, 16 * 16).to(model.fc1.weight.device)
        labels = labels.to(model.fc1.weight.device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total}%")

# Plotting the training loss and accuracy
epochs = range(1, n_epochs + 1)
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'r', label='Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, 'b', label='Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
