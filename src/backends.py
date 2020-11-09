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


class QiskitSimBackend:

    # TODO move to QasmUtils?
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
        backend_options = {"method": "statevector", "zero_threshold": config.floating_point_accuracy,
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

        ansatz_qasm = QiskitSimBackend.qasm_from_ansatz(ansatz, var_parameters)
        qasm += ansatz_qasm

        # Get a circuit of SWAP gates to reverse the order of qubits. This is required in order the statevector to
        # match the reversed order of qubits used by openfermion when obtaining the Hamiltonian Matrix.
        qasm.append(QasmUtils.reverse_qubits_qasm(n_qubits))

        qasm = ''.join(qasm)
        statevector = QiskitSimBackend.statevector_from_qasm(qasm)
        # print(qasm)
        return statevector

    @staticmethod
    def ham_sparse_matrix(q_system, excited_state=0):
        H_sparse_matrix = get_sparse_operator(q_system.jw_qubit_ham)
        if excited_state > 0:
            H_lower_state_terms = q_system.H_lower_state_terms
            assert H_lower_state_terms is not None
            assert len(H_lower_state_terms) >= excited_state
            for i in range(excited_state):
                term = H_lower_state_terms[i]
                state = term[1]
                statevector = QiskitSimBackend.statevector_from_ansatz(state.elements, state.parameters, state.n_qubits,
                                                                       state.n_electrons, init_state_qasm=state.init_state_qasm)
                # add the outer product of the lower lying state to the Hamiltonian
                H_sparse_matrix += scipy.sparse.csr_matrix(term[0] * numpy.outer(statevector, statevector))

        return H_sparse_matrix

    # return the expectation value of a qubit_operator
    @staticmethod
    def ham_expectation_value(var_parameters, ansatz, q_system, init_state_qasm=None, cache=None, excited_state=0):

        statevector = QiskitSimBackend.statevector_from_ansatz(ansatz, var_parameters, q_system.n_qubits,
                                                               q_system.n_electrons, init_state_qasm=init_state_qasm)
        sparse_statevector = scipy.sparse.csr_matrix(statevector)
        H_sparse_matrix = QiskitSimBackend.ham_sparse_matrix(q_system, excited_state=excited_state)

        expectation_value = \
            sparse_statevector.dot(H_sparse_matrix).dot(sparse_statevector.conj().transpose()).todense()[0, 0]

        return expectation_value.real

    # TODO check for excited states
    @staticmethod
    def ansatz_element_gradient(ansatz_element, var_parameters, ansatz, q_system,  cache=None, init_state_qasm=None, excited_state=0):

        excitations_generators = ansatz_element.excitations_generators
        exc_gen_sparse_matrices = []
        for exc_gen in excitations_generators:
            exc_gen_sparse_matrices.append(get_sparse_operator(exc_gen, n_qubits=q_system.n_qubits))
        exc_gen_sparse_matrix = sum(exc_gen_sparse_matrices)

        H_sparse_matrix = QiskitSimBackend.ham_sparse_matrix(q_system, excited_state=excited_state)

        commutator_sparse_matrix = H_sparse_matrix*exc_gen_sparse_matrix - exc_gen_sparse_matrix*H_sparse_matrix

        statevector = QiskitSimBackend.statevector_from_ansatz(ansatz, var_parameters, q_system.n_qubits,
                                                               q_system.n_electrons, init_state_qasm=init_state_qasm)
        sparse_statevector = scipy.sparse.csr_matrix(statevector)

        grad = sparse_statevector.dot(commutator_sparse_matrix).dot(sparse_statevector.conj().transpose()).todense()[0,0]

        assert grad.imag < config.floating_point_accuracy
        return grad.real

    # TODO check for excited states
    @staticmethod
    def ansatz_gradient(var_parameters, ansatz, q_system, cache=None, init_state_qasm=None, excited_state=0):

        assert len(ansatz) == len(var_parameters)
        ansatz_statevector = QiskitSimBackend.statevector_from_ansatz(ansatz, var_parameters, q_system.n_orbitals,
                                                                      q_system.n_electrons, init_state_qasm=init_state_qasm)
        ansatz_sparse_statevector = scipy.sparse.csr_matrix(ansatz_statevector)
        H_sparse_matrix = QiskitSimBackend.ham_sparse_matrix(q_system, excited_state=excited_state)

        phi = ansatz_sparse_statevector.transpose().conj()
        psi = H_sparse_matrix.dot(phi)

        ansatz_grad = []

        for i in range(len(ansatz))[::-1]:
            excitations_generators = ansatz[i].excitations_generators

            excitations_generators_matrices = []
            for term in excitations_generators:
                excitations_generators_matrices.append(get_sparse_operator(term, n_qubits=q_system.n_qubits))

            if len(excitations_generators_matrices) == 1:
                grad_i = 2 * (psi.transpose().conj().dot(excitations_generators_matrices[0]).dot(phi)).todense()[0, 0]

                ansatz_grad.append(grad_i.real)

                psi = scipy.sparse.linalg.expm_multiply(-var_parameters[i]*excitations_generators_matrices[0], psi)
                phi = scipy.sparse.linalg.expm_multiply(-var_parameters[i]*excitations_generators_matrices[0], phi)

            else:
                # TODO test properly
                assert len(excitations_generators_matrices) == 2

                psi = scipy.sparse.linalg.expm_multiply(-var_parameters[i] * excitations_generators_matrices[1], psi)
                phi = scipy.sparse.linalg.expm_multiply(-var_parameters[i] * excitations_generators_matrices[1], phi)

                grad_i = 2 * (psi.transpose().conj().dot(excitations_generators_matrices[0]+excitations_generators_matrices[1]).dot(phi)).todense()[0, 0]

                ansatz_grad.append(grad_i.real)

                psi = scipy.sparse.linalg.expm_multiply(-var_parameters[i] * excitations_generators_matrices[0], psi)
                phi = scipy.sparse.linalg.expm_multiply(-var_parameters[i] * excitations_generators_matrices[0], phi)

        ansatz_grad = ansatz_grad[::-1]
        # print(ansatz_grad)
        return numpy.array(ansatz_grad)


class MatrixCacheBackend:

    # return the expectation value of a qubit_operator
    @staticmethod
    def ham_expectation_value(var_parameters, ansatz,  q_system, cache, init_state_qasm=None, excited_state=0):

        sparse_statevector = cache.get_statevector(ansatz, list(var_parameters), init_state_qasm=init_state_qasm)
        H_sparse_matrix = cache.get_h_sparse_matrix()

        expectation_value = sparse_statevector.dot(H_sparse_matrix).dot(sparse_statevector.conj().transpose()).todense()[0, 0]

        return expectation_value.real

    # TODO check for excited states
    @staticmethod
    def ansatz_element_gradient(ansatz_element, var_parameters, ansatz, q_system, cache, init_state_qasm=None,
                                excited_state=0):

        sparse_statevector = cache.get_statevector(ansatz, list(var_parameters), init_state_qasm=init_state_qasm)
        commutator_sparse_matrix = cache.get_commutator_matrix(ansatz_element)

        grad = sparse_statevector.dot(commutator_sparse_matrix).dot(sparse_statevector.conj().transpose()).todense()[0, 0]

        assert grad.imag < config.floating_point_accuracy
        return grad.real

    # TODO check for excited states
    @staticmethod
    def ansatz_gradient(var_parameters, ansatz, q_system, cache, init_state_qasm=None, excited_state=0):

        assert len(ansatz) == len(var_parameters)
        ansatz_sparse_statevector = cache.get_statevector(ansatz, list(var_parameters), init_state_qasm=init_state_qasm)
        H_sparse_matrix = cache.H_sparse_matrix

        phi = ansatz_sparse_statevector.transpose().conj()
        psi = H_sparse_matrix.dot(phi)

        ansatz_grad = []

        for i in range(len(ansatz))[::-1]:
            excitations_generators = ansatz[i].excitations_generators

            excitations_generators_matrices = cache.exc_gen_sparse_matrices_dict[str(excitations_generators)]
            excitation_matrices = cache.get_ansatz_element_excitations_matrices(ansatz[i], var_parameters[i])
            # the variational parameter above should be with a minus. However this would required additional
            # calculations done by the cache. Avoid this by using that the exc. gen. is skew Hermitian and take the
            # conjugate transpose instead.
            excitation_matrices = [matrix.conj(copy=True).transpose() for matrix in excitation_matrices]

            if len(excitations_generators_matrices) == 1:
                grad_i = 2 * (psi.transpose().conj().dot(excitations_generators_matrices[0]).dot(phi)).todense()[
                    0, 0]

                ansatz_grad.append(grad_i.real)

                psi = excitation_matrices[0].dot(psi)
                phi = excitation_matrices[0].dot(phi)

            else:
                assert len(excitations_generators_matrices) == 2

                psi = excitation_matrices[1].dot(psi)
                phi = excitation_matrices[1].dot(phi)

                grad_i = 2 * (psi.transpose().conj().dot(
                    excitations_generators_matrices[0] + excitations_generators_matrices[1]).dot(phi)).todense()[
                    0, 0]

                ansatz_grad.append(grad_i.real)

                psi = excitation_matrices[0].dot(psi)
                phi = excitation_matrices[0].dot(phi)

        ansatz_grad = ansatz_grad[::-1]
        return numpy.array(ansatz_grad)
