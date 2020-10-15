from openfermion.transforms import get_fermion_operator, jordan_wigner, get_sparse_operator
from openfermion import QubitOperator
from openfermion.utils import jw_hartree_fock_state
import time

from src.utils import QasmUtils, MatrixUtils
from src import config

import qiskit.qasm
import scipy
import numpy
import logging


class QiskitSimCache:
    def __init__(self, q_system, qubit_operator=None):
        self.n_qubits = q_system.n_orbitals
        self.n_electrons = q_system.n_electrons
        self.jw_qubit_H = q_system.jw_qubit_ham
        # TODO check if storing this in multithreading leads to memory leak
        if qubit_operator is None:
            self.operator_sparse_matrix = get_sparse_operator(q_system.jw_qubit_ham)
        else:
            self.operator_sparse_matrix = get_sparse_operator(qubit_operator)
        self.statevector = None
        self.var_parameters = None

        # this is required for excited states calculations
        self.H_sparse_matrix_for_excited_state = None
        self.excite_state = None

    def update_statevector(self, ansatz, var_parameters, init_state_qasm=None):
        if self.var_parameters is not None and var_parameters == self.var_parameters:  # this condition is not neccesserily sufficinet
            assert self.statevector is not None
        else:
            self.var_parameters = var_parameters
            self.statevector = QiskitSim.statevector_from_ansatz(ansatz, var_parameters, self.n_qubits,
                                                                 self.n_electrons,
                                                                 init_state_qasm=init_state_qasm)
        return self.statevector

    # calculate the modified H for finding an excited state
    def calculate_hamiltonian_for_excited_state(self, H_lower_state_terms, excited_state=1):
        self.excite_state = excited_state
        if self.H_sparse_matrix_for_excited_state is not None:
            logging.warning('Calculating already existing variable in QiskitSimCache')

        self.H_sparse_matrix_for_excited_state = self.operator_sparse_matrix

        assert H_lower_state_terms is not None
        assert len(H_lower_state_terms) >= excited_state

        self.H_sparse_matrix_for_excited_state = QiskitSim.ham_sparse_matrix_for_exc_state(self.operator_sparse_matrix,
                                                                                  H_lower_state_terms[:excited_state])

        # return self.H_sparse_matrix_for_excited_state


class QiskitSim:

    @staticmethod
    def ham_sparse_matrix_for_exc_state(H_sparse_matrix, H_lower_state_terms):
        H_modified = H_sparse_matrix.copy()
        for term in H_lower_state_terms:
            state = term[1]
            statevector = QiskitSim.statevector_from_ansatz(state.ansatz, state.var_parameters, state.n_qubits,
                                                            state.n_electrons, init_state_qasm=state.init_state_qasm)
            # add the outer product of the lower lying state to the Hamiltonian
            H_modified += scipy.sparse.csr_matrix(term[0]*numpy.outer(statevector, statevector))

        return H_modified

    # get the qasm for an ansatz, defined by a list of ansatz elements (ansatz) and corresponding variational pars.
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

        ansatz_qasm = QiskitSim.qasm_from_ansatz(ansatz, var_parameters)
        qasm += ansatz_qasm

        # Get a circuit of SWAP gates to reverse the order of qubits. This is required in order the statevector to
        # match the reversed order of qubits used by openfermion when obtaining the Hamiltonian Matrix.
        qasm.append(QasmUtils.reverse_qubits_qasm(n_qubits))

        qasm = ''.join(qasm)
        statevector = QiskitSim.statevector_from_qasm(qasm)
        return statevector

    # return the expectation value of a qubit_operator
    @staticmethod
    def expectation_value(qubit_operator, ansatz, var_parameters, q_system, init_state_qasm=None, cache=None):

        if cache is None:
            statevector = QiskitSim.statevector_from_ansatz(ansatz, var_parameters, q_system.n_qubits,
                                                            q_system.n_electrons, init_state_qasm=init_state_qasm)
            sparse_statevector = scipy.sparse.csr_matrix(statevector)
            operator_sparse_matrix = get_sparse_operator(qubit_operator)
        else:
            statevector = cache.update_statevector(ansatz, list(var_parameters), init_state_qasm=init_state_qasm)
            sparse_statevector = scipy.sparse.csr_matrix(statevector)
            operator_sparse_matrix = cache.operator_sparse_matrix

        expectation_value = \
            sparse_statevector.dot(operator_sparse_matrix).dot(sparse_statevector.conj().transpose()).todense()[0, 0]

        return expectation_value.real

    @staticmethod
    def ham_expectation_value_exc_state(H_qubit_operator, ansatz, var_parameters, q_system, excited_state=1, cache=None,
                                        init_state_qasm=None):
        if cache is None:
            statevector = QiskitSim.statevector_from_ansatz(ansatz, var_parameters, q_system.n_qubits,
                                                            q_system.n_electrons, init_state_qasm=init_state_qasm)
            sparse_statevector = scipy.sparse.csr_matrix(statevector)
            operator_sparse_matrix = QiskitSim.ham_sparse_matrix_for_exc_state(get_sparse_operator(H_qubit_operator),
                                                                               q_system.H_lower_state_terms[:excited_state])
        else:
            statevector = cache.update_statevector(ansatz, list(var_parameters), init_state_qasm=init_state_qasm)
            sparse_statevector = scipy.sparse.csr_matrix(statevector)
            assert excited_state == cache.excite_state
            operator_sparse_matrix = cache.H_sparse_matrix_for_excited_state

        expectation_value =\
            sparse_statevector.dot(operator_sparse_matrix).dot(sparse_statevector.conj().transpose()).todense()[0, 0]

        return expectation_value

    # TODO check for excited states
    @staticmethod
    def excitation_gradient(excitation, ansatz, var_parameters, q_system, commutator_sparse_matrix=None,
                            init_state_qasm=None, cache=None, excited_state=0):

        if commutator_sparse_matrix is None:
            excitation_generator = excitation.excitation_generator
            assert type(excitation_generator) == QubitOperator

            exc_gen_sprs_m = get_sparse_operator(excitation_generator, n_qubits=q_system.n_qubits)

            if cache is None:
                if excited_state > 0:
                    H_sparse_matrix = QiskitSim.ham_sparse_matrix_for_exc_state(get_sparse_operator(q_system.jw_qubit_ham),
                                                                                q_system.H_lower_state_terms[:excited_state])
                else:
                    H_sparse_matrix = get_sparse_operator(q_system.jw_qubit_ham)
            else:
                if excited_state > 0:
                    H_sparse_matrix = cache.H_sparse_matrix_for_excited_state
                else:
                    H_sparse_matrix = cache.operator_sparse_matrix

            commutator_sparse_matrix = H_sparse_matrix*exc_gen_sprs_m - exc_gen_sprs_m*H_sparse_matrix

        if cache is None:
            statevector = QiskitSim.statevector_from_ansatz(ansatz, var_parameters, q_system.n_qubits,
                                                            q_system.n_electrons, init_state_qasm=init_state_qasm)
            sparse_statevector = scipy.sparse.csr_matrix(statevector)

        else:
            statevector = cache.update_statevector(ansatz, list(var_parameters), init_state_qasm=init_state_qasm)
            sparse_statevector = scipy.sparse.csr_matrix(statevector)

        grad = sparse_statevector.dot(commutator_sparse_matrix).dot(sparse_statevector.conj().transpose()).todense()[0,0]
        return grad

    # TODO check for excited states
    @staticmethod
    def ansatz_gradient(var_parameters, ansatz, q_system, init_state_qasm=None, cache=None, excited_state=0):
        """
            dE_i/dt_i = 2 Re{<psi_i| phi_i>}
        """

        assert len(ansatz) == len(var_parameters)
        if cache is None:
            ansatz_statevector = QiskitSim.statevector_from_ansatz(ansatz, var_parameters, q_system.n_orbitals,
                                                                   q_system.n_electrons, init_state_qasm=init_state_qasm)
            if excited_state > 0:
                H_sparse_matrix = QiskitSim.ham_sparse_matrix_for_exc_state(get_sparse_operator(q_system.jw_qubit_ham),
                                                                            q_system.H_lower_state_terms[:excited_state])
            else:
                H_sparse_matrix = get_sparse_operator(q_system.jw_qubit_ham)
        else:
            ansatz_statevector = cache.update_statevector(ansatz, list(var_parameters), init_state_qasm=init_state_qasm)
            if excited_state > 0:
                assert excited_state == cache.excite_state
                H_sparse_matrix = cache.H_sparse_matrix_for_excited_state
            else:
                H_sparse_matrix = cache.operator_sparse_matrix

        phi = scipy.sparse.csr_matrix(ansatz_statevector).transpose().conj()
        psi = H_sparse_matrix.dot(phi)

        ansatz_grad = []

        for i in range(len(ansatz))[::-1]:

            excitation_i_matrix = ansatz[i].excitation_generator_matrix
            if excitation_i_matrix is None:
                logging.warning('Try to supply a precomputed excitation_generator matrix, for faster execution')
                excitation_i_matrix = get_sparse_operator(ansatz[i].excitation_generator, n_qubits=q_system.n_qubits)

            grad_i = 2 * (psi.transpose().conj().dot(excitation_i_matrix).dot(phi)).todense()[0, 0]

            ansatz_grad.append(grad_i.real)

            psi = scipy.sparse.linalg.expm_multiply(-var_parameters[i]*excitation_i_matrix, psi)
            phi = scipy.sparse.linalg.expm_multiply(-var_parameters[i]*excitation_i_matrix, phi)

        ansatz_grad = ansatz_grad[::-1]

        return numpy.array(ansatz_grad)


# NOT USED
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

    # Hamiltonian expected value
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

