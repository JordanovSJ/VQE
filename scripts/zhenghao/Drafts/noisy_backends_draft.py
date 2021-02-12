from qiskit import Aer, execute
from scripts.zhenghao.Drafts.utils_li import *
from openfermion import QubitOperator


class NoisyBackend:

    # Evaluates expectation value of hamiltonian given in form of QubitOperator
    @staticmethod
    def ham_exp_value_swap(qasm_psi, op_ham: QubitOperator, n_shots=1024, noisy=False,
                           noise_model=None, coupling_map=None):
        assert op_ham.actions == ('X', 'Y', 'Z')
        # op_ham.terms returns a dictionary
        # The keys are tuples of the form ((1, 'X'), (2, 'Z'), (3, 'X'))
        # The values are the weight of each term
        ham_terms = op_ham.terms
        ham_terms_keys = list(ham_terms.keys())

        weighted_exp_val_list = []
        for gate in ham_terms_keys:
            qasm_U = QubitOpUtils.qasm_from_op_key(op_key=gate)
            weight = ham_terms[gate]
            exp_val_term = NoisyBackend.eval_exp_value_swap(qasm_psi=qasm_psi, qasm_U=qasm_U,
                                                            n_shots=n_shots, noisy=noisy,
                                                            noise_model=noise_model, coupling_map=coupling_map)
            weighted_exp_val_list.append(weight ** 2 * exp_val_term)

        return sum(weighted_exp_val_list)

    # Evaluates expectation value of a hamiltonian given in form of QubitOperator
    @staticmethod
    def ham_exp_value(qasm_psi, op_ham: QubitOperator, n_shots=1024, noisy=False,
                      noise_model=None, coupling_map=None):
        assert op_ham.actions == ('X', 'Y', 'Z')
        # op_ham.terms returns a dictionary
        # The keys are tuples of the form ((1, 'X'), (2, 'Z'), (3, 'X'))
        # The values are the weight of each term
        ham_terms = op_ham.terms
        ham_terms_keys = list(ham_terms.keys())

        weighted_exp_val_list = []
        for gate in ham_terms_keys:
            qasm_U = QubitOpUtils.qasm_from_op_key(op_key=gate)
            weight = ham_terms[gate]
            exp_val_term = NoisyBackend.eval_exp_value(qasm_psi=qasm_psi, qasm_U=qasm_U,
                                                       n_shots=n_shots, noisy=noisy,
                                                       noise_model=noise_model, coupling_map=coupling_map)
            weighted_exp_val_list.append(weight ** 2 * exp_val_term)

        return sum(weighted_exp_val_list)

    # Returns expectation value squared |<psi|U|psi>|^2=|<0|U_psi* U U_psi |0>|^2 by swap test
    @staticmethod
    def eval_exp_value_swap(qasm_psi, qasm_U, n_shots=1024, noisy=False,
                            noise_model=None, coupling_map=None):
        # The function can't deal with classical registers in qasm strings yet
        if 'creg' in qasm_psi:
            raise Exception('qasm_psi contains classical register')
        elif 'creg' in qasm_U:
            raise Exception('qasm_U contains classical register')

        if noisy and noise_model is None:
            raise Exception("No noise model")

        qasm_swap = QasmStrUtils.swap_test_qasm(qasm_psi=qasm_psi, qasm_U=qasm_U)

        if noisy:
            counts = NoisyBackend.noisy_sim(qasm_str=qasm_swap, noise_model=noise_model,
                                            coupling_map=coupling_map, n_shots=n_shots)
        else:
            counts = NoisyBackend.noiseless_sim(qasm_str=qasm_swap, n_shots=n_shots)

        # P(first qubit = 0) = 1/2 + 1/2* |<psi|U|psi>|^2
        exp_value = 2 * (counts['0'] / n_shots - 1 / 2)

        return exp_value

    # TO DO: What if initial state is not 00?
    # Returns expectation value squared |<psi|U|psi>|^2=|<0|U_psi* U U_psi |0>|^2
    @staticmethod
    def eval_exp_value(qasm_psi, qasm_U, n_shots=1024, noisy=False,
                       noise_model=None, coupling_map=None):

        # The function can't deal with classical registers in qasm strings yet
        if 'creg' in qasm_psi:
            raise Exception('qasm_psi contains classical register')
        elif 'creg' in qasm_U:
            raise Exception('qasm_U contains classical register')

        if noisy and noise_model is None:
            raise Exception("No noise model")

        # Construct qasm string for U_psi* U U_psi |0>
        qasm_psi_U = QasmStrUtils.qasm_append(qasm_psi, qasm_U)
        qasm_psi_invert = QasmStrUtils.qasm_invert(qasm_psi)
        qasm_psi_U_psi = QasmStrUtils.qasm_append(qasm_psi_U, qasm_psi_invert)

        # Add measurement to all qubits
        qasm_psi_U_psi = QasmStrUtils.qasm_measure_all(qasm_psi_U_psi)

        if noisy:
            counts = NoisyBackend.noisy_sim(qasm_str=qasm_psi_U_psi, noise_model=noise_model,
                                            coupling_map=coupling_map, n_shots=n_shots)
        else:
            counts = NoisyBackend.noiseless_sim(qasm_str=qasm_psi_U_psi, n_shots=n_shots)

        # THIS IS A BAD WAY OF RETURNING THE RESULT.
        # This is constructing a string of all 0s for number of qubits,
        # in order to compare with key value in counts
        n_qubits = QasmStrUtils.qreg_size_from_qasm(qasm_psi_U_psi)
        string_all0 = "0" * n_qubits
        if string_all0 in counts:
            exp_value = counts[string_all0] / n_shots
        else:
            exp_value = 0

        return exp_value

    # Construct circuit from qasm_str, run on qasm_simulator
    # Return counts as a dictionary
    @staticmethod
    def noiseless_sim(qasm_str: str, n_shots=1024):

        # construct quantum circuit
        circ = QuantumCircuit.from_qasm_str(qasm_str=qasm_str)

        # Select simulator to be Qasm from the Aer provider
        simulator = Aer.get_backend('qasm_simulator')

        # Execute under noiseless conditions and get counts
        result = execute(circ, simulator, shots=n_shots).result()
        counts = result.get_counts(circ)

        return counts

    # Construct circuit from qasm_str, run on qasm_simulator with noise_model
    # Return counts as a dictionary
    @staticmethod
    def noisy_sim(qasm_str: str, noise_model, coupling_map=None, n_shots=1024):

        # construct quantum circuit
        circ = QuantumCircuit.from_qasm_str(qasm_str=qasm_str)

        # Select simulator to be Qasm from the Aer provider
        simulator = Aer.get_backend('qasm_simulator')

        # Get basis gates for the noise model
        basis_gates = noise_model.basis_gates

        # Execute under noisy conditions and return counts
        result_noise = execute(circ, simulator, noise_model=noise_model,
                               basis_gates=basis_gates, coupling_map=coupling_map, shots=n_shots).result()
        counts_noise = result_noise.get_counts(circ)

        return counts_noise
