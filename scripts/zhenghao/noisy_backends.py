from openfermion.transforms import get_fermion_operator, jordan_wigner, get_sparse_operator
from openfermion import QubitOperator
from openfermion.utils import jw_hartree_fock_state
import time

from src.utils import QasmUtils, MatrixUtils
from src import config

from qiskit import QuantumCircuit, execute
from qiskit import Aer

import qiskit.qasm
import scipy
import numpy
import logging
from functools import partial


class QasmBackend:

    # Return the expectation value of a qubit operator
    @staticmethod
    def ham_expectation_value(var_parameters, ansatz, q_system, init_state_qasm=None,
                              n_shots=1024, noisy=False,
                              noise_model=None, coupling_map=None):
        """
        Returns expectation value of the q_system's hamiltonian with respect to the ansatz state
        defined by ansatz, parametrised by var_parameters

        :param init_state_qasm: qasm string for self customed initial state. If none, default is hf_state
        :param coupling_map: Coupling map for real device noise
        :param noisy: Is the evaluation noisy?
        :param n_shots: Number of shots of experiment
        :param noise_model: Noise model of the experiment
        :param var_parameters: List of variational parameters for the ansatz, same order as ansatz
        :param ansatz: List of ansatz elements that construct the ansatz circuit
        :param q_system: the molecule, whose hamiltonian expectation value we want to evaluate
        :return: the expectation value of the molecule's hamiltonian with respect to the ansatz state
        """

        # Do we need the noisy parameter? noise_model is None is enough to indicate noisy or not.
        if noisy and noise_model is None:
            raise Exception('Missing noise model')

        if init_state_qasm is None:
            init_state_qasm = QasmUtils.hf_state(q_system.n_electrons)
        qasm_ansatz = QasmBackend.qasm_from_ansatz(ansatz, var_parameters)
        qasm_psi = init_state_qasm + qasm_ansatz

        n_qubits = q_system.n_qubits

        hamiltonian = q_system.jw_qubit_ham  # The hamiltonian looks like h = Sum(h_i * tuple_i)
        ham_terms = hamiltonian.terms  # Returns a dictionary, tuple_i: h_i
        ham_keys_list = list(ham_terms.keys())  # List of tuple_i, each tuple_i looks like ((0, 'X'), (1, 'Z'))
        weights_list = list(ham_terms.values())  # List of h_i, each h_i is a complex number

        eval_expectation_value = partial(QasmBackend.eval_expectation_value, qasm_psi=qasm_psi,
                                         n_qubits=n_qubits, n_shots=n_shots, noisy=noisy,
                                         noise_model=noise_model, coupling_map=coupling_map)

        exp_value_list = [eval_expectation_value(op_U=operator) for operator in ham_keys_list]
        weighted_exp_value = [a * b for a, b in zip(exp_value_list, weights_list)]

        return sum(weighted_exp_value).real

    # Returns expectation value of <psi|U|psi> by manual pauli expectation evaluation
    # U is given in form of a tuple, e.g. ((1, X), (2, Z), (3, Y))
    @staticmethod
    def eval_expectation_value(qasm_psi: str, op_U: tuple, n_qubits, n_shots=1024, noisy=False,
                               noise_model=None, coupling_map=None):
        if op_U == ():
            return 1.0

        # Convert U into pauli expectation circuit
        qasm_pauli = QasmBackend.pauli_convert(QasmBackend.qasm_from_op_key(op_U))
        header = QasmUtils.qasm_header(n_qubits)

        qasm_to_eval = header + qasm_psi + qasm_pauli

        if noisy:
            counts = QasmBackend.noisy_sim(qasm_to_eval, noise_model=noise_model,
                                           coupling_map=coupling_map, n_shots=n_shots)
        else:
            counts = QasmBackend.noiseless_sim(qasm_to_eval, n_shots=n_shots)

        result_0 = '0' * n_qubits
        result_1 = '0' * (n_qubits - 1) + '1'
        if result_0 in counts:
            prob_0 = counts[result_0] / n_shots
        else:
            prob_0 = 0
        if result_1 in counts:
            prob_1 = counts[result_1] / n_shots
        else:
            prob_1 = 0

        return prob_0 - prob_1

    # Want to find the expectation value of <psi|U|psi>
    # Need to convert U into pauli measurement circuit
    # X = HZH, Y=SHZHS^dagger.
    # Replace Z strings with CNOT staircase
    # Measures last useful qubit onto c[0]
    @staticmethod
    def pauli_convert(qasm: str):
        assert qasm is not ''

        qasm_converted = ['']
        gate_list = qasm.splitlines()
        # the gate_list should look like
        # ['', 'y q[0];', 'z q[1];', 'z q[2];', 'y q[3];', 'x q[4];', 'x q[5];']
        # The gates can only be x,y,z. This is guaranteed by the qasm_from_op_key method

        previous_qubit = -1
        for gate in gate_list:
            if gate == '':
                continue

            ops = {
                'x': QasmBackend.x_gate_convert,
                'y': QasmBackend.y_gate_convert,
                'z': QasmBackend.z_gate_convert
            }

            current_qubit = QasmBackend.qubit_index_from_qasm(gate)

            chosen_operation_function = ops.get(gate[0])
            qasm_converted.append(chosen_operation_function(gate, previous_qubit, current_qubit))

            previous_qubit = current_qubit

        measure_gate = '\nmeasure q[{}] -> c[0];'.format(previous_qubit)
        qasm_converted.append(measure_gate)

        return ''.join(qasm_converted)

    @staticmethod
    def x_gate_convert(qasm_gate: str, previous_qubit: int, current_qubit: int):
        cnot_gate = ''
        if previous_qubit >= 0:
            cnot_gate = '\ncx q[{}],q[{}];'.format(previous_qubit, current_qubit)
        replace_gate = '\n' + qasm_gate.replace('x', 'h')
        return replace_gate + cnot_gate

    @staticmethod
    def y_gate_convert(qasm_gate: str, previous_qubit: int, current_qubit: int):
        cnot_gate = ''
        if previous_qubit >= 0:
            cnot_gate = '\ncx q[{}],q[{}];'.format(previous_qubit, current_qubit)
        replace_1 = '\n' + qasm_gate.replace('y', 'sdg')
        replace_2 = '\n' + qasm_gate.replace('y', 'h')
        return replace_1 + replace_2 + cnot_gate

    @staticmethod
    def z_gate_convert(qasm_gate: str, previous_qubit: int, current_qubit: int):
        cnot_gate = ''
        if previous_qubit >= 0:
            cnot_gate = '\ncx q[{}],q[{}];'.format(previous_qubit, current_qubit)
        return cnot_gate

    # Returns qubit index of a single qasm string
    # E.g. returns 1 from '\nz q[1];'
    @staticmethod
    def qubit_index_from_qasm(qasm: str):
        i = qasm.index('[')
        j = qasm.index(']')

        return int(qasm[i + 1:j])

    # Returns qasm string from ansatz, without header
    @staticmethod
    def qasm_from_ansatz(ansatz, var_parameters):
        qasm = ['']
        # perform ansatz operations
        n_used_var_pars = 0
        for element in ansatz:
            # take unused var. parameters for the ansatz element
            element_var_pars = var_parameters[n_used_var_pars:(n_used_var_pars + element.n_var_parameters)]
            n_used_var_pars += len(element_var_pars)
            qasm_element = element.get_qasm(element_var_pars)
            qasm.append(qasm_element)

        return ''.join(qasm)

    # Takes in argument of a tuple of the form ((1, 'X'), (2, 'Z'), (3, 'X')) or ()
    # Returns a qasm_string that implements the gate
    @staticmethod
    def qasm_from_op_key(op_key: tuple):
        qasm = ['']

        for gate in op_key:
            qubit_index = gate[0]
            gate_type = gate[1].lower()
            assert gate_type == 'x' or 'y' or 'z'
            qasm.append('\n{g_name} q[{q_index}];'.format(g_name=gate_type,
                                                          q_index=qubit_index))

        return ''.join(qasm)

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
