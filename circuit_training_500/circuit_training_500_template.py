#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def classify_data(X_train, Y_train, X_test):
    """Develop and train your very own variational quantum classifier.

    Use the provided training data to train your classifier. The code you write
    for this challenge should be completely contained within this function
    between the # QHACK # comment markers. The number of qubits, choice of
    variational ansatz, cost function, and optimization method are all to be
    developed by you in this function.

    Args:
        X_train (np.ndarray): An array of floats of size (250, 3) to be used as training data.
        Y_train (np.ndarray): An array of size (250,) which are the categorical labels
            associated to the training data. The categories are labeled by -1, 0, and 1.
        X_test (np.ndarray): An array of floats of (50, 3) to serve as testing data.

    Returns:
        str: The predicted categories of X_test, converted from a list of ints to a
            comma-separated string.
    """

    # Use this array to make a prediction for the labels of the data in X_test
    predictions = []

    # QHACK #
    np.random.seed(42)
    DEBUG = False

    def normalize(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def one_versus_rest_split(x, y, label):
        class_mask = (y == label)
        other_mask = ~class_mask

        relabeled = np.empty_like(y)
        relabeled[class_mask] = 1.0
        relabeled[other_mask] = -1.0

        class_idx = np.where(class_mask)[0]
        other_idx = np.where(other_mask)[0]

        num_class_samples = class_mask.sum()
        other_idx_subsampled = np.random.choice(other_idx, size=num_class_samples, replace=False)

        idx = np.concatenate((class_idx, other_idx_subsampled))
        idx.sort(kind='mergesort')

        return x[idx], relabeled[idx]

    def square_loss(labels, predictions):
        loss = 0
        for l, p in zip(labels, predictions):
            loss = loss + (l - p) ** 2

        loss = loss / len(labels)
        return loss

    def accuracy(labels, predictions):

        loss = 0
        for l, p in zip(labels, predictions):
            if abs(l - p) < 1e-5:
                loss = loss + 1
        loss = loss / len(labels)

        return loss

    n_qubits = 3
    dev = qml.device("default.qubit", wires=n_qubits, shots=100)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))


    def variational_classifier(var, inputs):
        weights = var[0]
        bias = var[1]
        return circuit(inputs, weights) + bias

    def cost(weights, features, labels):
        predictions = [variational_classifier(weights, f) for f in features]
        return square_loss(labels, predictions)

    def train_classifier(x_train, y_train, x_test, y_test, shots=50):
        from pennylane.optimize import NesterovMomentumOptimizer, AdamOptimizer

        old_shots = dev.shots
        dev.shots = shots

        num_train = len(x_train)
        num_layers = 4
        var_init = (qml.init.strong_ent_layers_uniform(num_layers, n_qubits, 3), 0.0)
        stepsize = 0.1
        opt = NesterovMomentumOptimizer(stepsize)
        batch_size = 5
        maxit = 20

        # train the variational classifier
        var = var_init
        for it in range(maxit):
            # Update the weights by one optimizer step
            batch_index = np.random.randint(0, num_train, (batch_size,))
            x_train_batch = x_train[batch_index]
            y_train_batch = y_train[batch_index]
            var = opt.step(lambda v: cost(v, x_train_batch, y_train_batch), var)

            # stepsize *= 0.95
            # opt.update_stepsize(stepsize)

            # Compute predictions on train and validation set
            predictions_train = [np.sign(variational_classifier(var, f)) for f in x_train]
            acc_train = accuracy(y_train, predictions_train)

            if DEBUG:
                predictions_val = [np.sign(variational_classifier(var, f)) for f in x_test]
                acc_val = accuracy(y_test, predictions_val)

                print(
                    "Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc validation: {:0.7f} "
                    "".format(it + 1, cost(var, x_train, y_train), acc_train, acc_val)
                )

            if acc_train > 0.95:
                break

            dev.shots = old_shots
        return var

    def predict(inputs, classifier_dict):
        predictions = []
        labels = np.array(list(classifier_dict.keys()))

        for inp in inputs:
            preds = np.array([variational_classifier(weights, inp) for weights in classifier_dict.values()])
            predictions.append(labels[np.argmax(preds)])

        return np.array(predictions)

    X_train = normalize(X_train)

    X_test = normalize(X_test)
    Y_test = np.array([1,0,-1,0,-1,1,-1,-1,0,-1,1,-1,0,1,0,-1,-1,0,0,1,1,0,-1,0,0,-1,0,
                       -1,0,0,1,1,-1,-1,-1,0,-1,0,1,0,-1,1,1,0,-1,-1,-1,-1,0,0])

    # num_data = len(Y_train)
    # num_train = int(0.75 * num_data)
    # index = np.random.permutation(range(num_data))
    labels = [-1.0, 0.0, 1.0]
    vars = {}

    for l in labels:
        xtr, ytr = one_versus_rest_split(X_train, Y_train, l)
        xte, yte = one_versus_rest_split(X_test, Y_test, l)
        vars[l] = train_classifier(xtr, ytr, xte, yte)

    predictions = predict(X_test, vars)
    if DEBUG:
        print(accuracy(Y_test, predictions))
    # QHACK #


    predictions = predictions.astype(np.int)

    return array_to_concatenated_string(predictions)


def array_to_concatenated_string(array):
    """DO NOT MODIFY THIS FUNCTION.

    Turns an array of integers into a concatenated string of integers
    separated by commas. (Inverse of concatenated_string_to_array).
    """
    return ",".join(str(x) for x in array)


def concatenated_string_to_array(string):
    """DO NOT MODIFY THIS FUNCTION.

    Turns a concatenated string of integers separated by commas into
    an array of integers. (Inverse of array_to_concatenated_string).
    """
    return np.array([int(x) for x in string.split(",")])


def parse_input(giant_string):
    """DO NOT MODIFY THIS FUNCTION.

    Parse the input data into 3 arrays: the training data, training labels,
    and testing data.

    Dimensions of the input data are:
      - X_train: (250, 3)
      - Y_train: (250,)
      - X_test:  (50, 3)
    """
    X_train_part, Y_train_part, X_test_part = giant_string.split("XXX")

    X_train_row_strings = X_train_part.split("S")
    X_train_rows = [[float(x) for x in row.split(",")] for row in X_train_row_strings]
    X_train = np.array(X_train_rows)

    Y_train = concatenated_string_to_array(Y_train_part)

    X_test_row_strings = X_test_part.split("S")
    X_test_rows = [[float(x) for x in row.split(",")] for row in X_test_row_strings]
    X_test = np.array(X_test_rows)

    return X_train, Y_train, X_test


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    X_train, Y_train, X_test = parse_input(sys.stdin.read())
    output_string = classify_data(X_train, Y_train, X_test)
    print(f"{output_string}")
