import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from circuit_training_500_template import parse_input


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


if __name__ == "__main__":
    X_train, Y_train, X_test = parse_input(sys.stdin.read())

    X_train = normalize(X_train)

    print(Y_train)
    print(X_train)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_train[:, 0],
               X_train[:, 1],
               X_train[:, 2],
               c=Y_train,
               cmap=matplotlib.colors.ListedColormap(['red', 'green', 'blue'])
    )

    plt.show()
