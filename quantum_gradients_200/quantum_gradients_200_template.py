#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def gradient_200(weights, dev):
    r"""This function must compute the gradient *and* the Hessian of the variational
    circuit using the parameter-shift rule, using exactly 51 device executions.
    The code you write for this challenge should be completely contained within
    this function between the # QHACK # comment markers.

    Args:
        weights (array): An array of floating-point numbers with size (5,).
        dev (Device): a PennyLane device for quantum circuit execution.

    Returns:
        tuple[array, array]: This function returns a tuple (gradient, hessian).

            * gradient is a real NumPy array of size (5,).

            * hessian is a real NumPy array of size (5, 5).
    """

    @qml.qnode(dev, interface=None)
    def circuit(w):
        for i in range(3):
            qml.RX(w[i], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RY(w[3], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RX(w[4], wires=2)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    gradient = np.zeros([5], dtype=np.float64)
    hessian = np.zeros([5, 5], dtype=np.float64)

    # QHACK #

    def first_oder_parameter_shift_term(qnode, params, i, shift):
        shifted = params.copy()
        shifted[np.unravel_index(i, shifted.shape)] += shift
        return qnode(shifted)

    def second_oder_parameter_shift_term(qnode, params, i, j, shift_i, shift_j):
        shifted = params.copy()
        shifted[np.unravel_index(i, shifted.shape)] += shift_i
        shifted[np.unravel_index(j, shifted.shape)] += shift_j
        return qnode(shifted)

    def parameter_shift(qnode, params, shift):
        gradient = np.zeros([5], dtype=np.float64)
        hessian = np.zeros([5, 5], dtype=np.float64)
        shift = 0.25 * np.pi

        for i in range(len(gradient)):
            for j in range(i + 1, len(gradient)):
                pp = second_oder_parameter_shift_term(qnode, params, i, j, shift, shift)
                mp = second_oder_parameter_shift_term(qnode, params, i, j, -shift, shift)
                pm = second_oder_parameter_shift_term(qnode, params, i, j, shift, -shift)
                mm = second_oder_parameter_shift_term(qnode, params, i, j, -shift, -shift)
                hij = (pp - mp - pm + mm) / (4 * np.sin(shift) ** 2)
                hessian[i, j] = hij
                hessian[j, i] = hij

        unshifted = qnode(params)

        for i in range(len(gradient)):
            p = first_oder_parameter_shift_term(qnode, params, i, 2 * shift)
            m = first_oder_parameter_shift_term(qnode, params, i, -2 * shift)
            gradient[i] = (p - m) / 2
            hessian[i, i] = (p - 2 * unshifted + m) / 2

        return gradient, hessian

    gradient, hessian = parameter_shift(circuit, weights, shift=0.25 * np.pi)

    # QHACK #

    return gradient, hessian, circuit.diff_options["method"]


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    weights = sys.stdin.read()
    weights = weights.split(",")
    weights = np.array(weights, float)

    dev = qml.device("default.qubit", wires=3)
    gradient, hessian, diff_method = gradient_200(weights, dev)

    print(
        *np.round(gradient, 10),
        *np.round(hessian.flatten(), 10),
        dev.num_executions,
        diff_method,
        sep=","
    )
