from openfermion.transforms import get_fermion_operator, jordan_wigner, get_sparse_operator
from openfermion import QubitOperator
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


class QiskitSimulation:

    @staticmethod
    def get_qasm_header(n_qubits):
        return 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{0}];\ncreg c[{0}];\n'.format(n_qubits)

    # get a qasm circuit for a qubit operator consisting of Pauli gates only (used for the Hamiltonian)
    @staticmethod
    def get_qubit_operator_qasm(qubit_operator):
        assert type(qubit_operator) == QubitOperator
        assert len(qubit_operator.terms) == 1

        operator = next(iter(qubit_operator.terms.keys()))
        coeff = next(iter(qubit_operator.terms.values()))

        assert coeff == 1

        # joining a list of string in a single step is more efficient than appending each time to a string (?)
        qasm = ['']

        for gate in operator:
            if gate[1] == 'X':
                qasm.append('x q[{0}];\n'.format(gate[0]))
            if gate[1] == 'Y':
                qasm.append('y q[{0}];\n'.format(gate[0]))
            if gate[1] == 'Z':
                qasm.append('z q[{0}];\n'.format(gate[0]))

        return ''.join(qasm)

    # return a qasm circuit for preparing the HF state for given number of qubits/orbitals and electrons, within JW
    @staticmethod
    def get_hf_state_qasm(n_qubits, n_electrons):
        qasm = ['']
        for i in range(n_electrons):
            qasm.append('x q[{0}];\n'.format(n_qubits - i))

        return ''.join(qasm)

    # returns a qasm circuit for an exponent of pauli operators
    @staticmethod
    def get_exponent_qasm(exponent_term, exponent_angle):
        assert type(exponent_term) == tuple  # TODO remove?
        assert exponent_angle.real == 0

        # gate to rotate the qubits to the corresponding basis (Z by default)
        x_basis_correction = ['']
        y_basis_correction_front = ['']
        y_basis_correction_back = ['']
        cnots = ['']

        for i, operator in enumerate(exponent_term):
            qubit = operator[0]
            pauli_operator = operator[1]

            # add basis rotations for X and Y
            if pauli_operator == 'X':
                x_basis_correction.append('h q[{}];\n'.format(qubit))
            if pauli_operator == 'Y':
                y_basis_correction_front.append('rx({}) q[{}];\n'.format(numpy.pi / 2, qubit))
                y_basis_correction_back.append('rx({}) q[{}];\n'.format(- numpy.pi / 2, qubit))

            # add the core cnot gates
            if i > 0:
                previous_qubit = exponent_term[i - 1][0]
                cnots.append('rx q[{}],q[{}];\n'.format(previous_qubit, qubit))

        front_basis_correction = x_basis_correction + y_basis_correction_front
        back_basis_correction = x_basis_correction + y_basis_correction_back

        # TODO make more readeble
        cnots_module = cnots.append('rz({}) q[{}];\n'.format(exponent_angle, exponent_term[-1][0])) + cnots[::-1]

        return ''.join(front_basis_correction + cnots_module + back_basis_correction)

    @staticmethod
    def get_excitation_list_qasm(excitation_list, excitation_parameters):
        qasm = ['']
        # iterate over all excitations (each excitation is represented by a sum of products of pauli operators)
        for i, excitation in enumerate(excitation_list):
            # iterate over the terms of each excitation (each term is a product of pauli operators, on different qubits)
            for exponent_term in excitation.terms:
                exponent_angle = excitation_parameters[i]*excitation.terms[exponent_term]
                qasm.append(QiskitSimulation.get_exponent_qasm(exponent_term, exponent_angle))

        return ''.join(qasm)

    # return a statevector in the form of an array from a qasm circuit
    @staticmethod
    def get_statevector_from_qasm(qasm_circuit):
        ansatz_circuit = qiskit.QuantumCircuit.from_qasm_str(qasm_circuit)
        backend = qiskit.BasicAer.get_backend('statevector_simulator')
        result = qiskit.execute(ansatz_circuit, backend).result()
        statevector = result.get_statevector(ansatz_circuit)

        return statevector

    @staticmethod
    def get_energy(qubit_hamiltonian, excitation_list, excitation_parameters, n_qubits, n_electrons, hf_initial_state=True):

        # add a qasm header
        qasm = [QiskitSimulation.get_qasm_header(n_qubits)]

        # add a circuit for HF state initialization
        if hf_initial_state:
            qasm.append(QiskitSimulation.get_hf_state_qasm(n_qubits, n_electrons))

        # add circuit elements implementing the list of excitations
        qasm.append(QiskitSimulation.get_excitation_list_qasm(excitation_list, excitation_parameters))

        # get the resulting statevector from the Qiskit simulator
        statevector = QiskitSimulation.get_statevector_from_qasm(qasm)

        # get the Hamiltonian in the form of a matrix
        hamiltonian_matrix = get_sparse_operator(qubit_hamiltonian).todense()

        return statevector.conj().dot(hamiltonian_matrix).dot(statevector)[0, 0]

# class RealDevice: