from openfermion.transforms import get_fermion_operator, jordan_wigner, get_sparse_operator
from openfermionpsi4 import run_psi4
from openfermion.hamiltonians import MolecularData
from openfermion.utils import jw_hartree_fock_state


# from qiskit.providers.aer import StatevectorSimulator
import qiskit

import scipy
import numpy


# # Parent class
# class Backend:

class MatrixCalculation:

    # NOT USED
    @staticmethod
    def get_sparse_statevector_module(sparse_statevector):
        return numpy.sqrt(sparse_statevector.conj().dot(sparse_statevector.transpose()).todense().item())

    # NOT USED
    @staticmethod
    def renormalize_sparse_statevector(sparse_statevector):
        statevector_module = numpy.sqrt(sparse_statevector.conj().dot(sparse_statevector.transpose()).todense().item())
        assert statevector_module.imag == 0
        return sparse_statevector / statevector_module

    # returns the compressed sparse row matrix for the exponent of a qubit operator
    @staticmethod
    def get_qubit_operator_exponent_sparse_matrix(qubit_operator, n_qubits, parameter=1):
        qubit_operator_matrix = get_sparse_operator(qubit_operator, n_qubits)
        return scipy.sparse.linalg.expm((parameter/2) * qubit_operator_matrix)

    @staticmethod
    def prepare_statevector(excitation_list, excitation_parameters, n_qubits, n_electrons, hf_initial_state=True):
        assert len(excitation_list) == len(excitation_parameters)
        assert n_qubits >= n_electrons

        # initiate statevector as the HF state or as the 0th state
        if hf_initial_state:
            sparse_statevector = scipy.sparse.csr_matrix(jw_hartree_fock_state(n_electrons, n_qubits))
        else:
            sparse_statevector = scipy.sparse.csr_matrix(numpy.zeros(2**n_qubits))

        for i, excitation in enumerate(excitation_list):
            excitation_matrix = MatrixCalculation.\
                get_qubit_operator_exponent_sparse_matrix(excitation, n_qubits, parameter=excitation_parameters[i])
            sparse_statevector = sparse_statevector.dot(excitation_matrix.transpose())  # TODO: is transpose needed?

            # renormalize
            # print('State vector module = ', self.get_sparse_vector_module(sparse_statevector))
            # sparse_statevector = self.renormalize_sparse_statevector(sparse_statevector)

        return sparse_statevector

    @staticmethod
    def get_energy(qubit_hamiltonian, excitation_list, excitation_parameters, n_qubits, n_electrons, hf_initial_state=True):
        sparse_matrix_hamiltonian = get_sparse_operator(qubit_hamiltonian)

        sparse_statevector = MatrixCalculation.\
            prepare_statevector(excitation_list, excitation_parameters, n_qubits, n_electrons,
                                hf_initial_state=hf_initial_state)
        bra = sparse_statevector.conj()
        ket = sparse_statevector.transpose()

        energy = bra.dot(sparse_matrix_hamiltonian).dot(ket)
        energy = energy.todense().item()

        return energy.real  # TODO: should we expect the energy to be real ?


class QiskitSimulator:

    @staticmethod
    def get_qasm_header(n_qubits):
        return 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{0}];\ncreg c[{0}];\n'.format(n_qubits)

    @staticmethod
    def get_excitation_circuit(excitation):

        return 'TODO'

    @staticmethod
    def get_hf_state_circuit(n_qubits, n_electrons):

        return 'TODO'

    @staticmethod
    def get_ansatz_circuit(excitation_list, excitation_parameters, n_qubits, n_electrons, hf_initial_state=True):
        # create a list containing the qasm elements
        qasm_circuit = [QiskitSimulator.get_qasm_header(n_qubits)]

        if hf_initial_state:
            dumy = 'dumy'

        qasm_circuit.append('x q[0];\n')
        return ''.join(qasm_circuit)

    @staticmethod
    def get_energy(qubit_hamiltonian, excitation_list, excitation_parameters, n_qubits, n_electrons, hf_initial_state=True):

        qasm_str = QiskitSimulator.get_ansatz_circuit(excitation_list, excitation_parameters, n_qubits, n_electrons)
        ansatz_circ = qiskit.QuantumCircuit.from_qasm_str(qasm_str)
        backend = qiskit.BasicAer.get_backend('statevector_simulator')
        result = qiskit.execute(ansatz_circ, backend).result()
        statevector = result.get_statevector(ansatz_circ)

        return statevector

# class RealDevice: