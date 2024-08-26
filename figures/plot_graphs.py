import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import os

# Add the parent directory to the system path to resolve the import issue
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from adversaries.bim2 import bim_attack
from adversaries.FGSM import fgsm_attack
from model.quantum_model import QModel
from dataset.mnist_dataset import get_mnist_dataloaders

def plot_clean_traintest(layers):
    plt.clf()

    train_losses = np.loadtxt('gen_data/clean/train_loss_{}layers.csv'.format(layers), delimiter=',')
    train_accuracies = np.loadtxt('gen_data/clean/train_accuracy_{}layers.csv'.format(layers), delimiter=',')
    test_losses = np.loadtxt('gen_data/clean/test_loss_{}layers.csv'.format(layers), delimiter=',')
    test_accuracies = np.loadtxt('gen_data/clean/test_accuracy_{}layers.csv'.format(layers), delimiter=',')

    avg_train_loss = train_losses[-1]
    avg_train_accuracy = train_accuracies[-1]
    avg_test_loss = test_losses[-1]
    avg_test_accuracy = test_accuracies[-1]

    epochs = len(avg_train_loss)

    # Plotting the training loss and accuracy
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    line1 = ax1.plot(range(1, epochs + 1, 10), avg_train_loss[::10], linestyle='-', marker='o', color='#16425B', label='Training loss')
    line2 = ax2.plot(range(1, epochs + 1, 10), avg_train_accuracy[::10], linestyle='-', marker='s', color='#75DDDD', label='Training accuracy')
    line3 = ax1.plot(range(1, epochs + 1, 10), avg_test_loss[::10], linestyle='-', marker='^', color='#16425B', label='Test loss')
    line4 = ax2.plot(range(1, epochs + 1, 10), avg_test_accuracy[::10], linestyle='-', marker='d', color='#75DDDD', label='Test accuracy')

    ax1.set_xlabel('Epochs', labelpad=10, fontweight='bold')
    ax1.set_ylabel('Loss', color='#16425B', labelpad=10, fontweight='bold')
    ax2.set_ylabel('Accuracy', color='#75DDDD', labelpad=10, fontweight='bold')

    lines = line1 + line2 + line3 + line4
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc='center right')
    fig.savefig('figures/clean_traintest_{}layers.png'.format(layers), bbox_inches='tight')

def plot_adv_accuracies(layers, attack, adv_ratio):
    plt.clf()

    train_accuracies = np.loadtxt('gen_data/{}/train_accuracy_{}layers_{}_advr{}.csv'.format(attack, layers, attack, int(adv_ratio * 100)), delimiter=',')
    test_accuracies = np.loadtxt('gen_data/{}/test_accuracy_{}layers_{}_advr{}.csv'.format(attack, layers, attack, int(adv_ratio * 100)), delimiter=',')
    adv_test_accuracies = np.loadtxt('gen_data/{}/adv_test_accuracy_{}layers_{}_advr{}.csv'.format(attack, layers, attack, int(adv_ratio * 100)), delimiter=',')

    avg_train_accuracy = train_accuracies[-1]
    avg_test_accuracy = test_accuracies[-1]
    avg_adv_test_accuracy = adv_test_accuracies[-1]

    epochs = len(avg_adv_test_accuracy)

    plt.plot(range(1, epochs + 1, 10), avg_train_accuracy[::10], label='Train accuracy')
    plt.plot(range(1, epochs + 1, 10), avg_test_accuracy[::10], label='Test accuracy - clean')
    plt.plot(range(1, epochs + 1, 10), avg_adv_test_accuracy[::10], label='Test accuracy - adversarial')
    plt.xlabel('Epochs', labelpad=10, fontweight='bold')
    plt.ylabel('Accuracy', labelpad=10, fontweight='bold')
    plt.legend()
    plt.savefig('figures/adv_accuracies_{}_{}layers_advr{}'.format(attack, layers, int(adv_ratio * 100)), bbox_inches='tight')

def plot_original_adversarial(attack, layers, adv_ratio, experiment_num):
    """
    str attack: 'bim' or 'fgsm'\n
    int layers: either 2 or 5\n
    float adv_ratio: between 0 and 1 inclusive, used only to determine model path\n
    int experiment_num: between 1 and 5 inclusive, used only to determine model path\n
    """
    plt.clf()

    model_path = 'train/weights/qModel{}_{}layers_clean_advr{}.pth'.format(experiment_num, layers, int(adv_ratio*100))
    model = QModel()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    _, test_loader = get_mnist_dataloaders()
    images, labels = next(iter(test_loader))

    loss = torch.nn.BCELoss()

    if attack == 'bim':
        adversaries = bim_attack(model, loss, images, labels, 0.3, 3, 1)
    elif attack == 'fgsm':
        adversaries = fgsm_attack(model, images, labels, 0.3, 1)

    outputs = model(adversaries)[:, 1]
    predictions = torch.round(outputs)

    cols, rows = 4, 2
    figure, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(4,8))
    for i in range(cols):
        figure.add_subplot(rows, cols, i+1)
        index = np.random.randint(0, len(adversaries)-1)
        plt.axis('off')
        plt.title("True: {}/Pred: {}".format(int(labels[index].item()), int(predictions[index].item())))
        plt.imshow(adversaries[index].detach().squeeze(), cmap='gray')
        figure.add_subplot(rows, cols, cols+i+1)
        plt.imshow(images[index].detach().squeeze(), cmap='gray')
        plt.axis('off')

    rowtitles = ['Adversarial Image', 'Original Image']

    for ax in axes.ravel():
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)

    for ax, row in zip(axes[:,0], rowtitles):
        ax.set_ylabel(row, size='x-large', fontweight='bold')
    
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(16, 9) # set figure's size manually to your full screen (32x18)
    figure.savefig('figures/original_adversarial.png', bbox_inches='tight')


plot_clean_traintest(5)
plot_adv_accuracies(5, 'fgsm', 0.3)
plot_original_adversarial('fgsm', 2, 1, 1)