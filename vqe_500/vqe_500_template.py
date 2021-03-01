#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def find_excited_states(H):
    """
    Fill in the missing parts between the # QHACK # markers below. Implement
    a variational method that can find the three lowest energies of the provided
    Hamiltonian.

    Args:
        H (qml.Hamiltonian): The input Hamiltonian

    Returns:
        The lowest three eigenenergies of the Hamiltonian as a comma-separated string,
        sorted from smallest to largest.
    """

    energies = np.zeros(3)

    # QHACK #
    dev = qml.device("default.qubit", wires=H.wires)
    np.random.seed(42)
    DEBUG = False

    def folded_spectrum_hamiltonian(H, energy_shift):
        from functools import reduce

        num_terms = len(H.coeffs)

        terms = []
        coeffs = []

        def get_newop(op):
            if isinstance(op.name, str):
                newop = getattr(qml, op.name)
                return newop(op.wires.labels)
            elif isinstance(op.name, list):
                newops = [getattr(qml, name)(wire) for name, wire in zip(op.name, op.wires.labels)]
                return reduce(lambda a, b: a @ b, newops)
            else:
                raise NotImplementedError

        for op, coeff in zip(H.ops, H.coeffs):
            coeffs.append(- 2 * energy_shift * coeff)
            terms.append(get_newop(op))

        for i in range(num_terms):
            for j in range(num_terms):
                op_i = get_newop(H.ops[i])
                op_j = get_newop(H.ops[j])

                if op_i.compare(op_j):
                    op = qml.Identity(0)
                else:
                    op = op_i @ op_j
                terms.append(op)
                coeffs.append(H.coeffs[i] * H.coeffs[j])

        terms.append(qml.Identity(0))
        coeffs.append(energy_shift ** 2)

        newH = qml.Hamiltonian(coeffs, terms, simplify=True)

        return newH

    def variational_ansatz(params, wires):
        qml.templates.StronglyEntanglingLayers(params, wires=wires)

    def run_vqe(H, ansatz, params=None):
        from pennylane.optimize import NesterovMomentumOptimizer, AdamOptimizer
        num_qubits = len(H.wires)
        num_layers = 4

        if params is None:
            params = qml.init.strong_ent_layers_uniform(num_layers, num_qubits, 3)

        cost_fn = qml.ExpvalCost(ansatz, H, dev)

        stepsize = 0.1
        opt = NesterovMomentumOptimizer(stepsize)
        max_iterations = 300
        conv_tol = 1e-8

        energy = 0

        for n in range(max_iterations):
            params, prev_energy = opt.step_and_cost(cost_fn, params)
            energy = cost_fn(params)
            conv = np.abs(energy - prev_energy)

            stepsize *= 0.99
            opt.update_stepsize(stepsize)

            if DEBUG and n % 20 == 0:
                print('Iteration = {:},  Energy = {:.8f} Ha'.format(n, energy))

            if conv <= conv_tol:
                break

        return energy, params

    def run_vqe_excited(H, ansatz, gs_params, params=None):
        from pennylane.optimize import NesterovMomentumOptimizer, AdamOptimizer
        num_qubits = len(H.wires)
        num_layers = 4

        if params is None:
            params = qml.init.strong_ent_layers_uniform(num_layers, num_qubits, 3)

        @qml.qnode(dev)
        def overlap(params, wires):
            variational_ansatz(gs_params, wires)
            qml.inv(qml.template(variational_ansatz)(params, wires))
            return qml.probs([0, 1, 2])

        def cost_fn(params, **kwargs):
            h_cost = qml.ExpvalCost(ansatz, H, dev)
            h = h_cost(params, **kwargs)
            o = overlap(params, wires=H.wires)
            return h + 1.5 * o[0]

        stepsize = 0.1
        opt = NesterovMomentumOptimizer(stepsize)
        max_iterations = 300
        conv_tol = 1e-8

        energy = 0

        for n in range(max_iterations):
            params, prev_energy = opt.step_and_cost(cost_fn, params)
            energy = cost_fn(params)
            conv = np.abs(energy - prev_energy)

            stepsize *= 0.99
            opt.update_stepsize(stepsize)

            if DEBUG and n % 20 == 0:
                print('Iteration = {:},  Energy = {:.8f} Ha'.format(n, energy))

            if conv <= conv_tol:
                break

        return energy, params

    def run_vqe_excited2(H, ansatz, gs_params, fes_params, params=None):
        from pennylane.optimize import NesterovMomentumOptimizer, AdamOptimizer
        num_qubits = len(H.wires)
        num_layers = 4

        if params is None:
            params = qml.init.strong_ent_layers_uniform(num_layers, num_qubits, 3)

        @qml.qnode(dev)
        def overlap(params, wires):
            variational_ansatz(gs_params, wires)
            qml.inv(qml.template(variational_ansatz)(params, wires))
            return qml.probs([0, 1, 2])

        @qml.qnode(dev)
        def overlap2(params, wires):
            variational_ansatz(fes_params, wires)
            qml.inv(qml.template(variational_ansatz)(params, wires))
            return qml.probs([0, 1, 2])

        def cost_fn(params, **kwargs):
            h_cost = qml.ExpvalCost(ansatz, H, dev)
            h = h_cost(params, **kwargs)
            o = overlap(params, wires=H.wires)
            o2 = overlap2(params, wires=H.wires)
            return h + 1.5 * o[0] + o2[0]

        stepsize = 0.3
        opt = NesterovMomentumOptimizer(stepsize)
        max_iterations = 300
        conv_tol = 1e-8

        energy = 0

        for n in range(max_iterations):
            params, prev_energy = opt.step_and_cost(cost_fn, params)
            energy = cost_fn(params)
            conv = np.abs(energy - prev_energy)

            stepsize *= 0.99
            opt.update_stepsize(stepsize)

            if DEBUG and n % 20 == 0:
                print('Iteration = {:},  Energy = {:.8f} Ha'.format(n, energy))

            if conv <= conv_tol:
                break

        return energy, params

    energy, params = run_vqe(H, variational_ansatz, params=None)
    if DEBUG:
        print(f"{energy:.6f}")

    energy2, params2 = run_vqe_excited(H, variational_ansatz, gs_params=params, params=None)
    if DEBUG:
        print(f"{energy2:.6f}")

    energy3, params3 = run_vqe_excited2(H, variational_ansatz, gs_params=params, fes_params=params2, params=None)
    if DEBUG:
        print(f"{energy3:.6f}")

    energies = [energy, energy2, energy3]
    if DEBUG:
        sol1 = np.array([-1.1658819, -1.0698449, -0.44972335])
        sol2 = np.array([-1.31795925, -0.99412998, -0.32243601])
        print(np.abs(energies-sol1)/sol1)
        print(np.abs(energies-sol2)/sol2)
    # QHACK #

    return ",".join([str(E) for E in energies])


def pauli_token_to_operator(token):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    Helper function to turn strings into qml operators.

    Args:
        token (str): A Pauli operator input in string form.

    Returns:
        A qml.Operator instance of the Pauli.
    """
    qubit_terms = []

    for term in token:
        # Special case of identity
        if term == "I":
            qubit_terms.append(qml.Identity(0))
        else:
            pauli, qubit_idx = term[0], term[1:]
            if pauli == "X":
                qubit_terms.append(qml.PauliX(int(qubit_idx)))
            elif pauli == "Y":
                qubit_terms.append(qml.PauliY(int(qubit_idx)))
            elif pauli == "Z":
                qubit_terms.append(qml.PauliZ(int(qubit_idx)))
            else:
                print("Invalid input.")

    full_term = qubit_terms[0]
    for term in qubit_terms[1:]:
        full_term = full_term @ term

    return full_term


def parse_hamiltonian_input(input_data):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    Turns the contents of the input file into a Hamiltonian.

    Args:
        filename(str): Name of the input file that contains the Hamiltonian.

    Returns:
        qml.Hamiltonian object of the Hamiltonian specified in the file.
    """
    # Get the input
    coeffs = []
    pauli_terms = []

    # Go through line by line and build up the Hamiltonian
    for line in input_data.split("S"):
        line = line.strip()
        tokens = line.split(" ")

        # Parse coefficients
        sign, value = tokens[0], tokens[1]

        coeff = float(value)
        if sign == "-":
            coeff *= -1
        coeffs.append(coeff)

        # Parse Pauli component
        pauli = tokens[2:]
        pauli_terms.append(pauli_token_to_operator(pauli))

    return qml.Hamiltonian(coeffs, pauli_terms)


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Turn input to Hamiltonian
    H = parse_hamiltonian_input(sys.stdin.read())

    # Send Hamiltonian through VQE routine and output the solution
    lowest_three_energies = find_excited_states(H)
    print(lowest_three_energies)
