#! /usr/bin/python3
import sys
import pennylane as qml
from pennylane import numpy as np

# DO NOT MODIFY any of these parameters
a = 0.7
b = -0.3
dev = qml.device("default.qubit", wires=3)


def natural_gradient(params):
    """Calculate the natural gradient of the qnode() cost function.

    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers.

    You should evaluate the metric tensor and the gradient of the QNode, and then combine these
    together using the natural gradient definition. The natural gradient should be returned as a
    NumPy array.

    The metric tensor should be evaluated using the equation provided in the problem text. Hint:
    you will need to define a new QNode that returns the quantum state before measurement.

    Args:
        params (np.ndarray): Input parameters, of dimension 6

    Returns:
        np.ndarray: The natural gradient evaluated at the input parameters, of dimension 6
    """

    natural_grad = np.zeros(6)

    # QHACK #
    grad_func = qml.grad(qnode)
    gradient = grad_func(params)[0]
    block_diag_mt = qml.metric_tensor(qnode)(params)
    approx_nat_gradient = np.dot(np.linalg.pinv(block_diag_mt), gradient)
    # print(gradient)
    # print(np.round(block_diag_mt, 8))
    # print(approx_nat_gradient)

    def second_oder_parameter_shift(params, i, j, shift_i, shift_j):
        shifted = params.copy()
        shifted[np.unravel_index(i, shifted.shape)] += shift_i
        shifted[np.unravel_index(j, shifted.shape)] += shift_j
        return shifted

    @qml.qnode(dev)
    def overlap(params, shifted_params):
        variational_circuit(shifted_params)
        qml.inv(qml.template(variational_circuit)(params))
        #obs = qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))
        return qml.probs([0, 1, 2])

    # print('\n', overlap(params, params)[0])

    f_matrix = np.zeros([6, 6], dtype=np.float64)
    shift = 0.5 * np.pi

    # print('\n', overlap(params, second_oder_parameter_shift(params, 1, 1, shift, shift))[0])

    for i in range(len(gradient)):
        for j in range(i, len(gradient)):
            pp = overlap(params, second_oder_parameter_shift(params, i, j, shift, shift))[0]
            mp = overlap(params, second_oder_parameter_shift(params, i, j, -shift, shift))[0]
            pm = overlap(params, second_oder_parameter_shift(params, i, j, shift, -shift))[0]
            mm = overlap(params, second_oder_parameter_shift(params, i, j, -shift, -shift))[0]
            fij = (-pp + mp + pm - mm) / 8.0
            f_matrix[i, j] = fij
            f_matrix[j, i] = fij

    # for i in range(len(gradient)):
    #     shifted = params.copy()
    #     shifted[np.unravel_index(i, shifted.shape)] += shift
    #     fii_prob = 1 - 0.5 * (overlap(params, shifted) + 1.0)
    #     f_matrix[i, i] = fii_prob - 1

    # print(np.round(f_matrix, 8))
    natural_grad = np.dot(np.linalg.pinv(f_matrix), gradient)

    # QHACK #


    return natural_grad


def non_parametrized_layer():
    """A layer of fixed quantum gates.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    qml.RX(a, wires=0)
    qml.RX(b, wires=1)
    qml.RX(a, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.RZ(a, wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(b, wires=1)
    qml.Hadamard(wires=0)


def variational_circuit(params):
    """A layered variational circuit composed of two parametrized layers of single qubit rotations
    interleaved with non-parameterized layers of fixed quantum gates specified by
    ``non_parametrized_layer``.

    The first parametrized layer uses the first three parameters of ``params``, while the second
    layer uses the final three parameters.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    non_parametrized_layer()
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)
    non_parametrized_layer()
    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)


@qml.qnode(dev)
def qnode(params):
    """A PennyLane QNode that pairs the variational_circuit with an expectation value
    measurement.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    variational_circuit(params)
    return qml.expval(qml.PauliX(1))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process inputs
    params = sys.stdin.read()
    params = params.split(",")
    params = np.array(params, float)

    updated_params = natural_gradient(params)

    print(*updated_params, sep=",")
