import torch
from dataset.mnist_dataset import get_data_loader  # Assumed function from mnist_dataset.py
from model.quantum_model import QuantumNet  # Assumed model import from quantum_model.py
from train import test_model  # Assumed testing function from train.py
import matplotlib.pyplot as plt

def fgsm_attack(model, data, target, epsilon):
    """Generate FGSM adversarial examples."""
    data.requires_grad = True
    output = model(data)
    model.zero_grad()
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output, target)
    loss.backward()
    data_grad = data.grad.data
    perturbed_data = data + epsilon * data_grad.sign()
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    return perturbed_data

def evaluate_attack(model, data_loader, epsilon_range):
    """Evaluate model on adversarial examples for different epsilons."""
    accuracies = []
    losses = []

    for epsilon in epsilon_range:
        correct = 0
        total = 0
        total_loss = 0
        for data, target in data_loader:
            perturbed_data = fgsm_attack(model, data, target, epsilon)
            output = model(perturbed_data)
            final_pred = output.max(1, keepdim=True)[1]
            correct += final_pred.eq(target.view_as(final_pred)).sum().item()
            loss = torch.nn.CrossEntropyLoss()(output, target)
            total_loss += loss.item()
            total += target.size(0)
        
        accuracy = 100. * correct / total
        average_loss = total_loss / total
        accuracies.append(accuracy)
        losses.append(average_loss)
        print(f'Epsilon: {epsilon:.2f} - Test Accuracy = {accuracy:.2f}% - Test Loss = {average_loss:.4f}')
    
    return accuracies, losses

# Load pretrained model and data
model = QuantumNet()
model.eval()
test_loader = get_data_loader()  # Assuming this function from mnist_dataset.py gives the test loader

# Range of epsilon values to test
epsilon_range = torch.arange(0, 0.35, 0.05)

# Evaluate the model on FGSM examples across a range of epsilons
accuracies, losses = evaluate_attack(model, test_loader, epsilon_range)

# Plotting
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epsilon_range, accuracies, 'o-')
plt.title('Accuracy vs Epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Accuracy (%)')

plt.subplot(1, 2, 2)
plt.plot(epsilon_range, losses, 'o-')
plt.title('Loss vs Epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()
