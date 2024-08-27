import numpy as np

def correct_average(attack, adv_ratio):

    #train_accuracy = np.loadtxt('gen_data/{}/adv_train_accuracy_5layers_{}_advr{}.csv'.format(attack, attack, int(adv_ratio * 100)), delimiter=',')
    #train_loss = np.loadtxt('gen_data/{}}/adv_train_loss_5layers_{}_advr{}.csv'.format(attack, attack, int(adv_ratio * 100)), delimiter=',')
    test_accuracy = np.loadtxt('gen_data/{}/adv_test_accuracy_5layers_{}_advr{}.csv'.format(attack, attack, int(adv_ratio * 100)), delimiter=',')
    test_loss = np.loadtxt('gen_data/{}/adv_test_loss_5layers_{}_advr{}.csv'.format(attack, attack, int(adv_ratio * 100)), delimiter=',')

    #raw_train_accuracy = train_accuracy[:5]
    #raw_train_loss = train_loss[:5]
    raw_test_accuracy = test_accuracy[:5]
    raw_test_loss = test_loss[:5]

    #raw_train_accuracy *= 2
    #raw_train_loss *= 2
    raw_test_accuracy *= 2
    raw_test_loss *= 2

    #avg_train_accuracy = np.mean(raw_train_accuracy, axis=0)
    #avg_train_loss = np.mean(raw_train_loss, axis=0)
    avg_test_accuracy = np.mean(raw_test_accuracy, axis=0)
    avg_test_loss = np.mean(raw_test_loss, axis=0)

    #train_accuracy[-1:] = avg_train_accuracy
    #train_loss[-1:] = avg_train_loss
    test_accuracy[-1:] = avg_test_accuracy
    test_loss[-1:] = avg_test_loss

    #np.savetxt('gen_data/{}/adv_train_accuracy_5layers_{}_advr{}.csv'.format(attack, attack, int(adv_ratio * 100)), train_accuracy, delimiter=',')
    #np.savetxt('gen_data/{}}/adv_train_loss_5layers_{}_advr{}.csv'.format(attack, attack, int(adv_ratio * 100)), train_loss, delimiter=',')
    np.savetxt('gen_data/{}/adv_test_accuracy_5layers_{}_advr{}.csv'.format(attack, attack, int(adv_ratio * 100)), test_accuracy, delimiter=',')
    np.savetxt('gen_data/{}/adv_test_loss_5layers_{}_advr{}.csv'.format(attack, attack, int(adv_ratio * 100)), test_loss, delimiter=',')

correct_average('fgsm', 0.3)
correct_average('fgsm', 1)