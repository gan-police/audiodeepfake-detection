"""Calculate mean and std of test and validation accuracies for different seeds."""
import pickle

import numpy as np


def calc_mean_std_accs(wavelet, mode, epochs):
    """Calculate mean and standard deviation of saved stats of training a model."""
    test_acc_lst = []
    val_acc_lst = []
    args = "8000_8736_4368_224_2000-4000_1_10000_melgan_0.001"
    for seed in range(7):
        matrix = pickle.load(
            open(
                f"/home/s6kogase/code/log/binary_{wavelet}_{args}_{epochs}e_{mode}_{seed}.pkl",
                "rb",
            )
        )[-1]
        # import pdb; pdb.set_trace()
        test_acc_lst.append(matrix["test_acc"])
        val_acc_lst.append(matrix["val_acc"][-1][-1])

    max = (np.max(test_acc_lst), np.max(val_acc_lst))
    mean = (np.mean(test_acc_lst), np.mean(val_acc_lst))
    std = (np.std(test_acc_lst), np.std(val_acc_lst))
    print(test_acc_lst)
    print(val_acc_lst)

    return max, mean, std


modes = ["onednet", "deeptestnet"]
epochs = [20, 5]
for i in range(len(modes)):
    print(f"{modes[i]}:")
    for wavelet in ["cmor4.6-0.87"]:
        max, mean, std = calc_mean_std_accs(wavelet, modes[i], epochs[i])
        print("test")
        print(
            f"{wavelet} & {100 * max[0]:.2f}% & {100 * mean[0]:.2f} +- {100 * std[0]:.2f}%"
        )
        print("val")
        print(
            f"{wavelet} & {100 * max[1]:.2f}% & {100 * mean[1]:.2f} +- {100 * std[1]:.2f}%"
        )
