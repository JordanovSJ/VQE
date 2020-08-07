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

    @staticmethod
    def get_expectation_value(q_system, ansatz_elements, var_parameters, operator_matrix=None,
                              initial_statevector_qasm=None, precomputed_statevector=None):

        t = time.time()
        # get the resulting statevector from the Qiskit simulator
        if precomputed_statevector is None:
            statevector, qasm = QiskitSim.statevector_from_ansatz(ansatz_elements, var_parameters,
                                                                  q_system.n_qubits, q_system.n_electrons,
                                                                  initial_statevector_qasm=initial_statevector_qasm)
        else:
            statevector = precomputed_statevector
            qasm = ''

        # get the operator in the form of a matrix
        if operator_matrix is None:
            operator_matrix = get_sparse_operator(q_system.jw_qubit_ham).todense()

        expectation_value = statevector.conj().dot(operator_matrix).dot(statevector)[0, 0]

        return expectation_value.real, statevector, qasm

    # NOT properly tested
    @staticmethod
    def ansatz_gradient(var_parameters, q_system, ansatz, init_state_qasm=None, ham_sparse_matrix=None,
                        precomputed_statevector=None):
        """
            dE_i/dt_i = 2 Re{<psi_i| phi_i>}
        """

        assert len(ansatz) == len(var_parameters)

        if precomputed_statevector is None:
            ansatz_statevector = QiskitSim.statevector_from_ansatz(ansatz, var_parameters, q_system.n_qubits,
                                                                   q_system.n_electrons,
                                                                   initial_statevector_qasm=init_state_qasm)[0]
        else:
            ansatz_statevector = precomputed_statevector

        if ham_sparse_matrix is None:
            ham_sparse_matrix = get_sparse_operator(q_system.jw_qubit_ham)

        phi = scipy.sparse.csr_matrix(ansatz_statevector).transpose().conj()
        psi = ham_sparse_matrix.dot(phi)

        ansatz_grad = []

        for i in range(len(ansatz))[::-1]:

            try:
                excitation_i_matrix = ansatz[i].excitation_matrix
            except AttributeError:
                excitation_i_matrix = get_sparse_operator(ansatz[i].excitation, n_qubits=q_system.n_qubits)

            grad_i = 2 * (psi.transpose().conj().dot(excitation_i_matrix).dot(phi)).todense()[0, 0]

            ansatz_grad.append(grad_i.real)

            psi = scipy.sparse.linalg.expm_multiply(-var_parameters[i]*excitation_i_matrix, psi)
            phi = scipy.sparse.linalg.expm_multiply(-var_parameters[i]*excitation_i_matrix, phi)

        ansatz_grad = ansatz_grad[::-1]

        return numpy.array(ansatz_grad)
