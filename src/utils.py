from openfermion.transforms import get_fermion_operator, jordan_wigner, get_sparse_operator
from openfermion import QubitOperator
from openfermion.utils import jw_hartree_fock_state
import time

import qiskit
import scipy
import numpy


class QasmUtils:

    @ staticmethod
    def gate_count(qasm, n_qubits):
        gate_counter = {}
        for i in range(n_qubits):
            # count all occurrences of a qubit (can get a few more because of the header)
            qubit_count = qasm.count('q[{}]'.format(i))
            cnot_count = qasm.count('q[{}],'.format(i))
            cnot_count += qasm.count(',q[{}]'.format(i))
            gate_counter['q{}'.format(i)] = {'cx': cnot_count, 'u1': qubit_count-cnot_count}
        return gate_counter

    @staticmethod
    def get_qasm_header(n_qubits):
        return 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{0}];\ncreg c[{0}];\n'.format(n_qubits)

    # NOT USED
    @staticmethod
    def get_pauli_operators_qasm(qubit_operator):
        assert type(qubit_operator) == QubitOperator
        assert len(qubit_operator.terms) == 1

        operator = next(iter(qubit_operator.terms.keys()))
        coeff = next(iter(qubit_operator.terms.values()))
        assert coeff == 1

        # represent the qasm as a list of strings
        qasm = ['']

        for gate in operator:
            qubit = gate[0]
            if gate[1] == 'X':
                qasm.append('x q[{0}];\n'.format(qubit))
            elif gate[1] == 'Y':
                qasm.append('y q[{0}];\n'.format(qubit))
            elif gate[1] == 'Z':
                qasm.append('z q[{0}];\n'.format(qubit))
            else:
                raise ValueError('Invalid qubit operator. {} is not a Pauli operator'.format(gate[1]))

        return ''.join(qasm)

    # return a qasm circuit for preparing the HF state
    @staticmethod
    def get_hf_state_qasm(n_electrons):
        qasm = ['']
        for i in range(n_electrons):
            qasm.append('x q[{0}];\n'.format(i))

        return ''.join(qasm)

    # get the qasm circuit of an excitation
    @staticmethod
    def get_excitation_qasm(excitation, var_parameter):
        qasm = ['']
        for exponent_term in excitation.terms:
            exponent_angle = var_parameter * excitation.terms[exponent_term]
            assert exponent_angle.real == 0
            exponent_angle = exponent_angle.imag
            qasm.append(QasmUtils.get_exponent_qasm(exponent_term, exponent_angle))

        return ''.join(qasm)

    # returns a qasm circuit for an exponent of pauli operators
    @staticmethod
    def get_exponent_qasm(exponent_term, exponent_angle):
        assert type(exponent_term) == tuple  # TODO remove?
        assert exponent_angle.imag == 0

        # gates for X and Y basis correction (Z by default)
        x_basis_correction = ['']
        y_basis_correction_front = ['']
        y_basis_correction_back = ['']
        # CNOT ladder
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
                cnots.append('cx q[{}],q[{}];\n'.format(previous_qubit, qubit))

        front_basis_correction = x_basis_correction + y_basis_correction_front
        back_basis_correction = x_basis_correction + y_basis_correction_back

        # TODO make this more readable
        # add a Z-rotation between the two CNOT ladders at the last qubit
        last_qubit = exponent_term[-1][0]
        z_rotation = 'rz({}) q[{}];\n'.format(-2*exponent_angle, last_qubit)  # exp(i*theta*Z) ~ Rz(-2*theta)

        # create the cnot module simulating a single Trotter step
        cnots_module = cnots + [z_rotation] + cnots[::-1]

        return ''.join(front_basis_correction + cnots_module + back_basis_correction)

    # get a circuit of SWAPs to reverse the order of qubits
    @staticmethod
    def reverse_qubits_qasm(n_qubits):
        qasm = ['']
        for i in range(int(n_qubits/2)):
            qasm.append('swap q[{}], q[{}];\n'.format(i, n_qubits - i - 1))

        return ''.join(qasm)


class MatrixUtils:
    # NOT USED
    @staticmethod
    def get_statevector_module(sparse_statevector):
        return numpy.sqrt(sparse_statevector.conj().dot(sparse_statevector.transpose()).todense().item())

    # NOT USED
    @staticmethod
    def renormalize_statevector(sparse_statevector):
        statevector_module = numpy.sqrt(sparse_statevector.conj().dot(sparse_statevector.transpose()).todense().item())
        assert statevector_module.imag == 0
        return sparse_statevector / statevector_module

    # returns the compressed sparse row matrix for the exponent of a qubit operator
    @staticmethod
    def get_qubit_operator_exponent_matrix(qubit_operator, n_qubits, parameter=1):
        assert parameter.imag == 0  # TODO remove?
        qubit_operator_matrix = get_sparse_operator(qubit_operator, n_qubits)
        return scipy.sparse.linalg.expm(parameter * qubit_operator_matrix)
