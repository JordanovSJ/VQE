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


class QiskitSim:

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

        ansatz_qasm = QiskitSim.qasm_from_ansatz(ansatz, var_parameters)
        qasm += ansatz_qasm

        # Get a circuit of SWAP gates to reverse the order of qubits. This is required in order the statevector to
        # match the reversed order of qubits used by openfermion when obtaining the Hamiltonian Matrix.
        qasm.append(QasmUtils.reverse_qubits_qasm(n_qubits))

        qasm = ''.join(qasm)
        statevector = QiskitSim.statevector_from_qasm(qasm)
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
                statevector = QiskitSim.statevector_from_ansatz(state.elements, state.parameters, state.n_qubits,
                                                                state.n_electrons, init_state_qasm=state.init_state_qasm)
                # add the outer product of the lower lying state to the Hamiltonian
                H_sparse_matrix += scipy.sparse.csr_matrix(term[0] * numpy.outer(statevector, statevector))

        return H_sparse_matrix

    # return the expectation value of a qubit_operator
    @staticmethod
    def ham_expectation_value(q_system, ansatz, var_parameters, init_state_qasm=None, cache=None, excited_state=0):

        if cache is None:
            statevector = QiskitSim.statevector_from_ansatz(ansatz, var_parameters, q_system.n_qubits,
                                                            q_system.n_electrons, init_state_qasm=init_state_qasm)
            sparse_statevector = scipy.sparse.csr_matrix(statevector)
            H_sparse_matrix = QiskitSim.ham_sparse_matrix(q_system, excited_state=excited_state)
        else:
            sparse_statevector = cache.update_statevector(ansatz, list(var_parameters), init_state_qasm=init_state_qasm)
            H_sparse_matrix = cache.H_sparse_matrix

        expectation_value = \
            sparse_statevector.dot(H_sparse_matrix).dot(sparse_statevector.conj().transpose()).todense()[0, 0]

        return expectation_value.real

    # TODO check for excited states
    @staticmethod
    def excitation_gradient(excitation, ansatz, var_parameters, q_system, init_state_qasm=None, cache=None, excited_state=0):

        if cache is None:
            excitation_generator = excitation.excitation_generator
            assert type(excitation_generator) == QubitOperator
            exc_gen_sparse_matrix = get_sparse_operator(excitation_generator, n_qubits=q_system.n_qubits)
            H_sparse_matrix = QiskitSim.ham_sparse_matrix(q_system, excited_state=excited_state)
            commutator_sparse_matrix = H_sparse_matrix*exc_gen_sparse_matrix - exc_gen_sparse_matrix*H_sparse_matrix

            statevector = QiskitSim.statevector_from_ansatz(ansatz, var_parameters, q_system.n_qubits,
                                                            q_system.n_electrons, init_state_qasm=init_state_qasm)
            sparse_statevector = scipy.sparse.csr_matrix(statevector)
        else:
            sparse_statevector = cache.update_statevector(ansatz, list(var_parameters), init_state_qasm=init_state_qasm)
            commutator_sparse_matrix = cache.commutator_sparse_matrices[str(excitation.excitation_generator)]

        grad = sparse_statevector.dot(commutator_sparse_matrix).dot(sparse_statevector.conj().transpose()).todense()[0,0]
        assert grad.imag < config.floating_point_accuracy
        return grad.real

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
            ansatz_sparse_statevector = scipy.sparse.csr_matrix(ansatz_statevector)
            H_sparse_matrix = QiskitSim.ham_sparse_matrix(q_system, excited_state=excited_state)
        else:
            ansatz_sparse_statevector = cache.update_statevector(ansatz, list(var_parameters), init_state_qasm=init_state_qasm)
            H_sparse_matrix = cache.H_sparse_matrix

        phi = ansatz_sparse_statevector.transpose().conj()
        psi = H_sparse_matrix.dot(phi)

        ansatz_grad = []

        for i in range(len(ansatz))[::-1]:

            if cache is None:
                # logging.warning('Try to supply a precomputed excitation_generator matrix, for faster execution')
                excitation_i_matrix = get_sparse_operator(ansatz[i].excitation_generator, n_qubits=q_system.n_qubits)
            else:
                excitation_i_matrix = cache.exc_gen_sparse_matrices[str(ansatz[i].excitation_generator)]

            grad_i = 2 * (psi.transpose().conj().dot(excitation_i_matrix).dot(phi)).todense()[0, 0]

            ansatz_grad.append(grad_i.real)

            psi = scipy.sparse.linalg.expm_multiply(-var_parameters[i]*excitation_i_matrix, psi)
            phi = scipy.sparse.linalg.expm_multiply(-var_parameters[i]*excitation_i_matrix, phi)

        ansatz_grad = ansatz_grad[::-1]
        return numpy.array(ansatz_grad)


# # Excitations statevector simulator: super efficient for states constructed by excitations only
# class ExcStateSim:
#
#     @staticmethod
#     def hf_statevector(n_qubits, n_electrons):
#         statevector = numpy.zeros(2 ** n_qubits)
#         # MAGIC
#         hf_term = 0
#         for i in range(n_electrons):
#             hf_term += 2 ** (n_qubits - 1 - i)
#         statevector[hf_term] = 1
#         return statevector
#
#     # return a statevector in the form of an array from a list of ansatz elements
#     @staticmethod
#     def statevector_from_ansatz(ansatz, var_parameters, n_qubits, n_electrons, cache, init_state_qasm=None):
#         assert cache is not None
#         statevector = ExcStateSim.hf_statevector(n_qubits, n_electrons)
#         sparse_statevector = scipy.sparse.csr_matrix(statevector).transpose()
#         identity = scipy.sparse.identity(2 ** n_qubits)
#
#         for i, excitation in enumerate(ansatz):
#             parameter = var_parameters[i]
#             term1 = numpy.sin(parameter)*cache.get_exc_gen_sparse_matrix(excitation)
#             term2 = (1 - numpy.cos(parameter))*cache.get_sqr_exc_gen_sparse_matrix(excitation)
#             excitation_matrix = identity + term1 + term2
#             sparse_statevector = excitation_matrix.dot(sparse_statevector)
#         statevector = numpy.array(sparse_statevector.transpose().todense())[0]
#         return statevector
#
#     @staticmethod
#     def ham_sparse_matrix(q_system, cache, excited_state=0):
#         H_sparse_matrix = get_sparse_operator(q_system.jw_qubit_ham)
#         if excited_state > 0:
#             H_lower_state_terms = q_system.H_lower_state_terms
#             assert H_lower_state_terms is not None
#             assert len(H_lower_state_terms) >= excited_state
#             for i in range(excited_state):
#                 term = H_lower_state_terms[i]
#                 state = term[1]
#                 statevector = ExcStateSim.statevector_from_ansatz(state.elements, state.parameters, state.n_qubits,
#                                                                   state.n_electrons, cache,
#                                                                   init_state_qasm=state.init_state_qasm)
#                 # add the outer product of the lower lying state to the Hamiltonian
#                 H_sparse_matrix += scipy.sparse.csr_matrix(term[0] * numpy.outer(statevector, statevector))
#
#         return H_sparse_matrix
#
#     # return the expectation value of a qubit_operator
#     @staticmethod
#     def ham_expectation_value(q_system, ansatz, var_parameters, init_state_qasm=None, cache=None, excited_state=0):
#         print('koi')
#         assert cache is not None
#
#         if cache is None:
#             statevector = ExcStateSim.statevector_from_ansatz(ansatz, var_parameters, q_system.n_qubits,
#                                                               q_system.n_electrons, cache,
#                                                               init_state_qasm=init_state_qasm)
#             sparse_statevector = scipy.sparse.csr_matrix(statevector)
#             H_sparse_matrix = ExcStateSim.ham_sparse_matrix(q_system, cache, excited_state=excited_state)
#         else:
#             sparse_statevector = cache.update_statevector(ansatz, list(var_parameters), init_state_qasm=init_state_qasm,
#                                                           backend=ExcStateSim)
#             H_sparse_matrix = cache.H_sparse_matrix
#
#         expectation_value = \
#             sparse_statevector.dot(H_sparse_matrix).dot(sparse_statevector.conj().transpose()).todense()[0, 0]
#
#         return expectation_value.real
#
#     # TODO check for excited states
#     @staticmethod
#     def excitation_gradient(excitation, ansatz, var_parameters, q_system, init_state_qasm=None, cache=None,
#                             excited_state=0):
#
#         if cache is None:
#             excitation_generator = excitation.excitation_generator
#             assert type(excitation_generator) == QubitOperator
#             exc_gen_sparse_matrix = get_sparse_operator(excitation_generator, n_qubits=q_system.n_qubits)
#             H_sparse_matrix = ExcStateSim.ham_sparse_matrix(q_system, cache,  excited_state=excited_state)
#             commutator_sparse_matrix = H_sparse_matrix * exc_gen_sparse_matrix - exc_gen_sparse_matrix * H_sparse_matrix
#
#             statevector = ExcStateSim.statevector_from_ansatz(ansatz, var_parameters, q_system.n_qubits,
#                                                               q_system.n_electrons, cache,
#                                                               init_state_qasm=init_state_qasm)
#             sparse_statevector = scipy.sparse.csr_matrix(statevector)
#         else:
#             sparse_statevector = cache.update_statevector(ansatz, list(var_parameters), init_state_qasm=init_state_qasm,
#                                                           backend=ExcStateSim)
#             commutator_sparse_matrix = cache.commutator_sparse_matrices[str(excitation.excitation_generator)]
#
#         grad = sparse_statevector.dot(commutator_sparse_matrix).dot(sparse_statevector.conj().transpose()).todense()[
#             0, 0]
#         assert grad.imag < config.floating_point_accuracy
#         return grad.real
#
#     # TODO check for excited states
#     @staticmethod
#     def ansatz_gradient(var_parameters, ansatz, q_system, init_state_qasm=None, cache=None, excited_state=0):
#         """
#             dE_i/dt_i = 2 Re{<psi_i| phi_i>}
#         """
#
#         assert len(ansatz) == len(var_parameters)
#         if cache is None:
#             ansatz_statevector = ExcStateSim.statevector_from_ansatz(ansatz, var_parameters, q_system.n_orbitals,
#                                                                      q_system.n_electrons, cache,
#                                                                      init_state_qasm=init_state_qasm)
#             ansatz_sparse_statevector = scipy.sparse.csr_matrix(ansatz_statevector)
#             H_sparse_matrix = ExcStateSim.ham_sparse_matrix(q_system, cache,  excited_state=excited_state)
#         else:
#             ansatz_sparse_statevector = cache.update_statevector(ansatz, list(var_parameters),
#                                                                  init_state_qasm=init_state_qasm, backend=ExcStateSim)
#             H_sparse_matrix = cache.H_sparse_matrix
#
#         phi = ansatz_sparse_statevector.transpose().conj()
#         psi = H_sparse_matrix.dot(phi)
#
#         ansatz_grad = []
#
#         for i in range(len(ansatz))[::-1]:
#
#             if cache is None:
#                 # logging.warning('Try to supply a precomputed excitation_generator matrix, for faster execution')
#                 excitation_i_matrix = get_sparse_operator(ansatz[i].excitation_generator, n_qubits=q_system.n_qubits)
#             else:
#                 excitation_i_matrix = cache.exc_gen_sparse_matrices[str(ansatz[i].excitation_generator)]
#
#             grad_i = 2 * (psi.transpose().conj().dot(excitation_i_matrix).dot(phi)).todense()[0, 0]
#
#             ansatz_grad.append(grad_i.real)
#
#             psi = scipy.sparse.linalg.expm_multiply(-var_parameters[i] * excitation_i_matrix, psi)
#             phi = scipy.sparse.linalg.expm_multiply(-var_parameters[i] * excitation_i_matrix, phi)
#
#         ansatz_grad = ansatz_grad[::-1]
#         return numpy.array(ansatz_grad)

