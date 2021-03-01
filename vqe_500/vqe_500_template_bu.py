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

    def variational_ansatz(params, wires=H.wires):
        n_qubits = len(wires)
        n_rotations = len(params)

        if n_rotations > 1:
            n_layers = n_rotations // n_qubits
            n_extra_rots = n_rotations - n_layers * n_qubits

            # Alternating layers of unitary rotations on every qubit followed by a
            # ring cascade of CNOTs.
            for layer_idx in range(n_layers):
                layer_params = params[layer_idx * n_qubits: layer_idx * n_qubits + n_qubits, :]
                qml.broadcast(qml.Rot, wires, pattern="single", parameters=layer_params)
                qml.broadcast(qml.CNOT, wires, pattern="ring")

            # There may be "extra" parameter sets required for which it's not necessarily
            # to perform another full alternating cycle. Apply these to the qubits as needed.
            extra_params = params[-n_extra_rots:, :]
            extra_wires = wires[: n_qubits - 1 - n_extra_rots: -1]
            qml.broadcast(qml.Rot, extra_wires, pattern="single", parameters=extra_params)
        else:
            # For 1-qubit case, just a single rotation to the qubit
            qml.Rot(*params[0], wires=wires[0])

    def run_vqe(H, ansatz, params=None, gs_params=None):
        num_qubits = len(H.wires)
        num_param_sets = (2 ** num_qubits) - 1

        if params is None:
            params = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=(num_param_sets, 3))

        H_cost_fn = qml.ExpvalCost(ansatz, H, dev)
        cost_fn = H_cost_fn

        if gs_params is not None:
            @qml.qnode(dev)
            def overlap(params, wires):
                variational_ansatz(gs_params, wires)
                qml.inv(qml.template(variational_ansatz)(params, wires))
                # obs = qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))
                return qml.probs([0, 1, 2])

            cost_fn = lambda p: H_cost_fn(p) + overlap(p, gs_params)[0]

        opt = qml.AdagradOptimizer(stepsize=0.4)

        max_iterations = 500
        conv_tol = 1e-6

        energy = 0

        for n in range(max_iterations):
            params, prev_energy = opt.step_and_cost(cost_fn, params)
            energy = cost_fn(params)
            conv = np.abs(energy - prev_energy)

            if n % 20 == 0:
                print('Iteration = {:},  Energy = {:.8f} Ha'.format(n, energy))

            if conv <= conv_tol:
                break

        return energy, params

    energy, params = run_vqe(H, variational_ansatz, params=None)
    print(f"{energy:.6f}")

    energy2, params2 = run_vqe(H, variational_ansatz, params=params, gs_params=params)
    print(f"{energy2:.6f}")

    # shifted_energy = -0.44972335#energy + 0.05
    # H2 = folded_spectrum_hamiltonian(H, shifted_energy)
    # H2a = H @ H - 2 * shifted_energy * H + shifted_energy ** 2 * qml.Identity(0)
    # H2b = (H - shifted_energy * qml.Identity(0)) @ (H - shifted_energy * qml.Identity(0))
    # energy2, params = run_vqe(H2, variational_ansatz, params=params)
    # print(f"{energy2:.6f}")

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
