import torch
import os
import sys
import random

# Add the parent directory to the system path to resolve the import issue
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from dataset.mnist_dataset import get_mnist_dataloaders  # Assumed function from mnist_dataset.py
from model.quantum_model import QModel  # Assumed model import from quantum_model.py
import matplotlib.pyplot as plt

def fgsm_batch(model, batchdata, epsilon):
    for data, target in batchdata:
        perturbed_data = fgsm_attack(model, data, target, epsilon)
    return perturbed_data

def fgsm_attack(model, data, target, epsilon):
    """Generate FGSM adversarial examples."""
    data.requires_grad = True
    output = model(data)[:, 1]
    target = torch.Tensor.float(target)
    model.zero_grad()
    criterion = torch.nn.BCELoss()
    #print(output.dtype, target.dtype)
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
    originals = []
    perturbed = []

    cols, rows = 7, 1
    figure, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(14,2))

    for i, epsilon in enumerate(epsilon_range):
        correct = 0
        total = 0
        total_loss = 0
        for data, target in data_loader:
            originals.append(data)
            perturbed_data = fgsm_attack(model, data, target, epsilon)
            perturbed.append(perturbed_data)
            output = model(perturbed_data)[:, 1]
            target = torch.Tensor.float(target)
            #print(output.dtype, target.dtype)
            final_pred = torch.round(output)
            correct += final_pred.eq(target.view_as(final_pred)).sum().item()
            loss = torch.nn.BCELoss()(output, target)
            total_loss += loss.item()
            total += target.size(0)
        
        accuracy = 100. * correct / total
        average_loss = total_loss / total
        accuracies.append(accuracy)
        losses.append(average_loss)
        print(f'Epsilon: {epsilon:.2f} - Test Accuracy = {accuracy:.2f}% - Test Loss = {average_loss:.4f}')
        figure.add_subplot(rows, cols, i+1)
        index = random.randint(0, len(perturbed)-1)
        plt.axis('off')
        plt.title("epsilon = {:.2f}".format(epsilon))
        plt.imshow(perturbed[index].detach().squeeze(), cmap='gray')
        
    plt.show()
    return accuracies, losses

def main():
    # Load pretrained model and data
    model = QModel()
    model.eval()
    _, test_loader = get_mnist_dataloaders(batch_size=256)  # Assuming this function from mnist_dataset.py gives the test loader
    '''
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
    '''
    image = fgsm_batch(model, test_loader, epsilon=0.1)
    plt.imshow(image[0].detach().squeeze(), cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()