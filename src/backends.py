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


class QiskitSim:
    def __init__(self, q_system, store_H_sparse_matrx=True):
        # self.q_system = q_system
        self.n_qubits = q_system.n_orbitals
        self.n_electrons = q_system.n_electrons
        self.jw_qubit_H = q_system.jw_qubit_ham
        if store_H_sparse_matrx:
            self.H_sparse_matrix = get_sparse_operator(q_system.jw_qubit_ham)
        else:
            self.H_sparse_matrix = None
        self.statevector = None
        self.var_parameters = None

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
    def statevector_from_ansatz(ansatz, var_parameters, n_qubits, n_electrons, init_state_qasm=None):
        assert n_electrons < n_qubits
        qasm = ['']
        qasm.append(QasmUtils.qasm_header(n_qubits))

        # initial state
        if init_state_qasm is None:
            qasm.append(QasmUtils.hf_state(n_electrons))
        else:
            qasm.append(init_state_qasm)

        ansatz_qasm = QasmUtils.ansatz_qasm(ansatz, var_parameters)
        qasm += ansatz_qasm

        # Get a circuit of SWAP gates to reverse the order of qubits. This is required in order the statevector to
        # match the reversed order of qubits used by openfermion when obtaining the Hamiltonian Matrix. This is not
        # required in the case of implementing the H as a circuit (when running on a real device)
        qasm.append(QasmUtils.reverse_qubits_qasm(n_qubits))

        qasm = ''.join(qasm)

        statevector = QiskitSim.statevector_from_qasm(qasm)

        return statevector

    def get_updated_statevector(self, ansatz, var_parameters, init_state_qasm=None):
        if self.var_parameters is not None and var_parameters == self.var_parameters:  # this condition is not neccesserily sufficinet
            assert self.statevector is not None
        else:
            self.var_parameters = var_parameters
            self.statevector = QiskitSim.statevector_from_ansatz(ansatz, var_parameters, self.n_qubits, self.n_electrons,
                                                                 init_state_qasm=init_state_qasm)

        return self.statevector

    def get_expectation_value(self, ansatz, var_parameters, operator_sparse_matrix=None, init_state_qasm=None):

        statevector = self.get_updated_statevector(ansatz, list(var_parameters), init_state_qasm=init_state_qasm)
        sparse_statevector = scipy.sparse.csr_matrix(statevector)

        # get the operator in the form of a sparse matrix
        if operator_sparse_matrix is None:
            operator_sparse_matrix = self.H_sparse_matrix

        expectation_value = \
            sparse_statevector.dot(operator_sparse_matrix).dot(sparse_statevector.conj().transpose()).todense()[0, 0]

        return expectation_value.real  #, statevector

    def get_excitation_gradient(self, excitation, ansatz, var_parameters, commutator_sparse_matrix=None,
                                init_state_qasm=None, update_statevector=True):
        if commutator_sparse_matrix is None:
            excitation_generator = excitation.excitation_generator
            assert type(excitation) == QubitOperator
            commutator_sparse_matrix =\
                get_sparse_operator(self.jw_qubit_H * excitation_generator - excitation_generator * self.jw_qubit_H,
                                    n_qubits=self.n_qubits)
        if update_statevector:
            statevector = self.get_updated_statevector(ansatz, list(var_parameters), init_state_qasm=init_state_qasm)
            sparse_statevector = scipy.sparse.csr_matrix(statevector)
        else:
            sparse_statevector = scipy.sparse.csr_matrix(self.statevector)
        return sparse_statevector.dot(commutator_sparse_matrix).dot(sparse_statevector.conj().transpose()).todense()[0, 0]

    # NOT properly tested
    def get_ansatz_gradient(self, var_parameters, ansatz, init_state_qasm=None):
        """
            dE_i/dt_i = 2 Re{<psi_i| phi_i>}
        """

        assert len(ansatz) == len(var_parameters)
        ansatz_statevector = self.get_updated_statevector(ansatz, list(var_parameters), init_state_qasm=init_state_qasm)
        phi = scipy.sparse.csr_matrix(ansatz_statevector).transpose().conj()
        psi = self.H_sparse_matrix.dot(phi)

        ansatz_grad = []

        for i in range(len(ansatz))[::-1]:

            excitation_i_matrix = ansatz[i].excitation_matrix
            if excitation_i_matrix is None:
                print('Compute excitation matrix')  # TEST
                excitation_i_matrix = get_sparse_operator(ansatz[i].excitation_generator, n_qubits=self.n_qubits)

            grad_i = 2 * (psi.transpose().conj().dot(excitation_i_matrix).dot(phi)).todense()[0, 0]

            ansatz_grad.append(grad_i.real)

            psi = scipy.sparse.linalg.expm_multiply(-var_parameters[i]*excitation_i_matrix, psi)
            phi = scipy.sparse.linalg.expm_multiply(-var_parameters[i]*excitation_i_matrix, phi)

        ansatz_grad = ansatz_grad[::-1]

        return numpy.array(ansatz_grad)


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
                get_excitation_matrix(ansatz_element.excitation_generator, n_qubits, parameter=var_parameters[i])
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

