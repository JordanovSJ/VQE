from openfermion.transforms import get_fermion_operator, jordan_wigner, get_sparse_operator
from openfermion import QubitOperator
from openfermion.utils import jw_hartree_fock_state
import time

from src.utils import QasmUtils, MatrixUtils

import qiskit
import qiskit.qasm
import scipy
import numpy


class MatrixCalculation:

    @staticmethod
    def prepare_statevector(ansatz_elements, var_parameters, n_qubits, n_electrons, initial_statevector=None):
        assert len(ansatz_elements) == len(var_parameters)
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
                get_qubit_operator_exponent_matrix(ansatz_element.excitation, n_qubits, parameter=var_parameters[i])
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


class QiskitSimulation:

    # return a statevector in the form of an array from a qasm circuit
    @staticmethod
    def get_statevector_from_qasm(qasm_circuit):
        n_threads = 2
        backend_options = {"method": "statevector", "zero_threshold": 10e-9, "max_parallel_threads": n_threads,
                           "max_parallel_experiments": n_threads, "max_parallel_shots": n_threads}
        backend = qiskit.Aer.get_backend('statevector_simulator')
        qiskit_circuit = qiskit.QuantumCircuit.from_qasm_str(qasm_circuit)
        result = qiskit.execute(qiskit_circuit, backend, backend_options=backend_options).result()
        statevector = result.get_statevector(qiskit_circuit)
        return statevector

    # return a statevector in the form of an array from a list of ansatz elements
    @staticmethod
    def get_statevector_from_ansatz_elements(ansatz_elements, var_parameters, n_qubits, n_electrons, initial_statevector_qasm=None):
        assert n_electrons < n_qubits
        qasm = ['']
        qasm.append(QasmUtils.qasm_header(n_qubits))

        # initial state
        if initial_statevector_qasm is None:
            qasm.append(QasmUtils.hf_state(n_electrons))
        else:
            qasm.append(initial_statevector_qasm)

        # perform ansatz operations
        n_used_var_pars = 0
        for element in ansatz_elements:
            # take unused var. parameters for the ansatz element
            element_var_pars = var_parameters[n_used_var_pars:(n_used_var_pars + element.n_var_parameters)]
            n_used_var_pars += len(element_var_pars)
            qasm_element = element.get_qasm(element_var_pars)
            qasm.append(qasm_element)

        # Get a circuit of SWAP gates to reverse the order of qubits. This is required in order the statevector to
        # match the reversed order of qubits used by openfermion when obtaining the Hamiltonian Matrix. This is not
        # required in the case of implementing the H as a circuit as well (when running on a real device)
        qasm.append(QasmUtils.reverse_qubits_qasm(n_qubits))

        qasm = ''.join(qasm)

        statevector = QiskitSimulation.get_statevector_from_qasm(qasm)

        return statevector, qasm

    @staticmethod
    def get_energy(qubit_hamiltonian, ansatz_elements, var_parameters, n_qubits, n_electrons, initial_statevector_qasm=None):

        # get the resulting statevector from the Qiskit simulator
        statevector, qasm = QiskitSimulation.get_statevector_from_ansatz_elements(ansatz_elements, var_parameters,
                                                                                  n_qubits, n_electrons,
                                                                                  initial_statevector_qasm=initial_statevector_qasm)

        # get the Hamiltonian in the form of a matrix
        hamiltonian_matrix = get_sparse_operator(qubit_hamiltonian).todense()
        energy = statevector.conj().dot(hamiltonian_matrix).dot(statevector)[0, 0]

        return energy.real, statevector, qasm

    @staticmethod
    def get_exponent_energy_gradient(qubit_hamiltonian, exponent_term, ansatz_elements, var_parameters, n_qubits, n_electrons,
                            initial_statevector_qasm=None):

        assert type(exponent_term) == QubitOperator
        exponent_matrix = get_sparse_operator(exponent_term, n_qubits).todense()

        assert type(qubit_hamiltonian) == QubitOperator
        hamiltonian_matrix = get_sparse_operator(qubit_hamiltonian).todense()

        statevector, qasm = QiskitSimulation.get_statevector_from_ansatz_elements(ansatz_elements, var_parameters,
                                                                                  n_qubits, n_electrons,
                                                                                  initial_statevector_qasm=initial_statevector_qasm)

        commutator = hamiltonian_matrix.dot(exponent_matrix) - exponent_matrix.dot(hamiltonian_matrix)
        gradient = statevector.conj().dot(commutator).dot(statevector)[0, 0]

        return gradient.real


