from openfermion.transforms import get_fermion_operator, jordan_wigner, get_sparse_operator
from openfermion import QubitOperator
from openfermion.utils import jw_hartree_fock_state
import time

from src.utils import QasmUtils, MatrixUtils
from src import config

import qiskit.qasm
import scipy
import numpy
import ray


class MatrixCalculation:

    @staticmethod
    def prepare_statevector(ansatz_elements, var_parameters, n_qubits, n_electrons, initial_statevector=None):
        assert len(ansatz_elements) == len(var_parameters)  # TODO this is wrong
        assert n_qubits >= n_electrons

        # initiate statevector as the HF state or as the 0th state
        if initial_statevector is None:
            sparse_statevector = scipy.sparse.csr_matrix(jw_hartree_fock_state(n_electrons, n_qubits))
        else:
            assert len(initial_statevector) == 2**n_qubits
            assert initial_statevector.dot(initial_statevector.conj()) == 1
            sparse_statevector = scipy.sparse.csr_matrix(initial_statevector)

        for i, ansatz_element in enumerate(ansatz_elements):
            # assert ansatz_element.element_type == 'excitation'
            excitation_matrix = MatrixUtils.\
                get_excitation_matrix(ansatz_element.excitation, n_qubits, parameter=var_parameters[i])
            sparse_statevector = sparse_statevector.dot(excitation_matrix.transpose())

        return sparse_statevector

    @staticmethod
    def get_energy(qubit_hamiltonian, ansatz_elements, var_parameters, n_qubits, n_electrons, initial_statevector=None):

        sparse_matrix_hamiltonian = get_sparse_operator(qubit_hamiltonian)

        sparse_statevector = MatrixCalculation.\
            prepare_statevector(ansatz_elements, var_parameters, n_qubits, n_electrons,
                                initial_statevector=initial_statevector)
        bra = sparse_statevector.conj()
        ket = sparse_statevector.transpose()

        energy = bra.dot(sparse_matrix_hamiltonian).dot(ket)
        energy = energy.todense().item()

        statevector = numpy.array(sparse_statevector.todense())[0]

        return energy, statevector, None


class QiskitSim:

    # return a statevector in the form of an array from a qasm circuit
    @staticmethod
    def statevector_from_qasm(qasm_circuit):
        n_threads = config.qiskit_n_threads
        backend_options = {"method": "statevector", "zero_threshold": config.qiskit_zero_threshold,
                           "max_parallel_threads": n_threads, "max_parallel_experiments": n_threads,
                           "max_parallel_shots": n_threads}
        backend = qiskit.Aer.get_backend('statevector_simulator')
        qiskit_circuit = qiskit.QuantumCircuit.from_qasm_str(qasm_circuit)

        result = qiskit.execute(qiskit_circuit, backend, backend_options=backend_options).result()
        statevector = result.get_statevector(qiskit_circuit)
        return statevector

    # return a statevector in the form of an array from a list of ansatz elements
    @staticmethod
    def statevector_from_ansatz(ansatz_elements, var_parameters, n_qubits, n_electrons,
                                initial_statevector_qasm=None):
        assert n_electrons < n_qubits
        qasm = ['']
        qasm.append(QasmUtils.qasm_header(n_qubits))

        # initial state
        if initial_statevector_qasm is None:
            qasm.append(QasmUtils.hf_state(n_electrons))
        else:
            qasm.append(initial_statevector_qasm)

        ansatz_qasm = QasmUtils.ansatz_qasm(ansatz_elements, var_parameters)
        qasm += ansatz_qasm

        # Get a circuit of SWAP gates to reverse the order of qubits. This is required in order the statevector to
        # match the reversed order of qubits used by openfermion when obtaining the Hamiltonian Matrix. This is not
        # required in the case of implementing the H as a circuit (when running on a real device)
        qasm.append(QasmUtils.reverse_qubits_qasm(n_qubits))

        qasm = ''.join(qasm)

        statevector = QiskitSim.statevector_from_qasm(qasm)

        return statevector, qasm

    # TODO add q_system and use q_system.matrix_jw_ham
    @staticmethod
    def get_expectation_value(q_system, ansatz_elements, var_parameters, operator_matrix=None,
                              initial_statevector_qasm=None):

        t = time.time()
        # get the resulting statevector from the Qiskit simulator
        statevector, qasm = QiskitSim.statevector_from_ansatz(ansatz_elements, var_parameters,
                                                              q_system.n_qubits, q_system.n_electrons,
                                                              initial_statevector_qasm=initial_statevector_qasm)
        # get the operator in the form of a matrix
        if operator_matrix is None:
            operator_matrix = q_system.dense_matrix_jw_ham

        expectation_value = statevector.conj().dot(operator_matrix).dot(statevector)[0, 0]

        return expectation_value.real, statevector, qasm

    @staticmethod
    def ansatz_gradient(var_parameters, q_system, ansatz, init_state_qasm=None, ansatz_statevector=None):
        assert len(ansatz) == len(var_parameters)

        if ansatz_statevector is None:
            ansatz_statevector = QiskitSim.statevector_from_ansatz(ansatz, var_parameters, q_system.n_qubits,
                                                                   q_system.n_electrons, initial_statevector_qasm=init_state_qasm)[0]

        # t0 = time.time()

        ansatz_grad = []
        # transformed_ham = q_system.dense_matrix_jw_ham
        transformed_ham = q_system.sparse_matrix_jw_ham

        for i in range(len(ansatz))[::-1]:

            # t1 = time.time()
            transformed_statevector = QiskitSim.statevector_from_ansatz(ansatz[:i], var_parameters[:i], q_system.n_qubits,
                                                                        q_system.n_electrons, initial_statevector_qasm=init_state_qasm)[0]
            # print('dt 1 {}'.format(time.time() - t1))

            # t1 = time.time()
            excitation_i_matrix = get_sparse_operator(ansatz[i].excitation, n_qubits=q_system.n_qubits).todense()
            # print('dt 2 {}'.format(time.time() - t1))

            # t1 = time.time()
            grad_i = 2*ansatz_statevector.conj().dot(transformed_ham.todense()).dot(excitation_i_matrix).dot(transformed_statevector)[0, 0].round(15)
            # print('dt 3 {}'.format(time.time() - t1))

            assert grad_i.imag == 0

            ansatz_grad.append(grad_i.real)

            # t1 = time.time()
            ansatz_element_matrix = MatrixUtils.get_excitation_matrix(ansatz[i].excitation, q_system.n_qubits,
                                                                      var_parameters[i])
            # print('dt 4 {}'.format(time.time() - t1))

            # t1 = time.time()
            transformed_ham = transformed_ham.dot(ansatz_element_matrix)
            # print('dt 5 {}'.format(time.time() - t1))

            # print('grad element {}'.format(i))
            # print('dt = {}'.format(time.time()-t0))
            # t0 = time.time()

        ansatz_grad = ansatz_grad[::-1]

        return numpy.array(ansatz_grad)

    # Not used
    @staticmethod
    def ansatz_excitation_gradient(q_system, ansatz, var_parameters, excitation_index, ansatz_statevector=None,
                                   init_state_qasm=None):

        if init_state_qasm is None:
            init_state_qasm = QasmUtils.hf_state(q_system.n_electrons)

        if ansatz_statevector is None:
            ansatz_statevector = numpy.array(
                QiskitSim.statevector_from_ansatz(ansatz, var_parameters, q_system.n_qubits, q_system.n_electrons,
                                                  initial_statevector_qasm=init_state_qasm)[0]
            )

        hamiltonian_matrix = q_system.dense_matrix_jw_ham
        if hamiltonian_matrix is None:
            hamiltonian_matrix = get_sparse_operator(q_system.jw_qubit_ham).todense()

        # build the 2nd statevector
        grad_statevector = numpy.zeros(len(ansatz_statevector)) + 0j  # force grad_statevector to be stored as a complex array

        qasm_first = QasmUtils.qasm_header(q_system.n_qubits) + init_state_qasm
        qasm_first += QasmUtils.ansatz_qasm(ansatz[:excitation_index+1], var_parameters[:excitation_index+1])
        qasm_last = QasmUtils.ansatz_qasm(ansatz[excitation_index+1:], var_parameters[excitation_index+1:])

        excitation_operator = ansatz[excitation_index].excitation
        for pauli_word in excitation_operator.get_operators():
            coefficient = list(pauli_word.terms.values())[0]
            assert coefficient.real == 0  # it should be skew-Hermitian
            pauli_word_qasm = QasmUtils.pauli_word_qasm(pauli_word / coefficient)
            grad_statevector_qasm = qasm_first + pauli_word_qasm + qasm_last + QasmUtils.reverse_qubits_qasm(q_system.n_qubits)
            grad_statevector += coefficient*QiskitSim.statevector_from_qasm(grad_statevector_qasm)

        gradient = ansatz_statevector.conj().dot(hamiltonian_matrix).dot(grad_statevector)[0, 0]*2  # TODO check

        return gradient.real
