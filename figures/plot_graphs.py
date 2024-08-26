import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add the parent directory to the system path to resolve the import issue
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

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


plot_clean_traintest(5)
plot_adv_accuracies(5, 'fgsm', 0.3)