# this module contains functions not designated a better place
import scipy
import numpy
import openfermion
from src.utils import QasmUtils
from src.ansatz_elements import AnsatzElement


def prepare_statevector_as_matrix(excitations_list, initial_statevector):

    statevector = initial_statevector
    # n_qubits = len(statevector)

    for excitation in excitations_list:
        operator, parameter = excitation

        # operator_matrix = openfermion.transforms.get_sparse_operator(operator, n_qubits)
        statevector = scipy.sparse.linalg.expm_multiply(-1j*parameter*operator, statevector)

    return statevector


# get a list of compressed sparse row matrices, corresponding to the excitation list, including the var. params
def get_excitation_matrix_list(self, params):

        assert len(self.excitation_list) == len(params)

        excitation_matrix_list = []
        for i, excitation in enumerate(self.excitation_list):
            excitation_matrix_list.append(self.get_qubit_operator_exponent_matrix(params[i]*excitation))

        return excitation_matrix_list


def get_excitation_list_qasm(excitation_list, var_parameters, gate_counter):
    qasm = ['']
    # iterate over all excitations (each excitation is represented by a sum of products of pauli operators)
    for i, excitation in enumerate(excitation_list):
        # print('Excitation ', i)  # testing
        # iterate over the terms of each excitation (each term is a product of pauli operators, on different qubits)
        # TODO replace with the function from QasmUtils
        for exponent_term in excitation.terms:
            exponent_angle = var_parameters[i] * excitation.terms[exponent_term]
            assert exponent_angle.real == 0
            exponent_angle = exponent_angle.imag
            qasm.append(QasmUtils.exponent_qasm(exponent_term, exponent_angle))

    return ''.join(qasm)


# does not work
class HeuristicAnsatz1:
    def __init__(self, n_orbitals, n_electrons):
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons
        self.ansatz_type = 'hardware_efficient'

    def get_element_qasm(self, double_parameters=False):

        qasm_middle = ['']
        qasm_cnots_odd = ['']
        qasm_cnots_even = ['']

        # apply single qubit general rotations to each qubit
        for qubit in range(0, self.n_orbitals - 1):
            if qubit % 2:
                qasm_cnots_odd.append('cx q[{}], q[{}];\n'.format(qubit, qubit + 1))
            else:
                qasm_cnots_even.append('cx q[{}], q[{}];\n'.format(qubit, qubit + 1))
            if double_parameters:
                qasm_middle.append('rx({{}}) q[{}];\n'.format(qubit))  # we want to leave first {} empty for var_parameter later
                qasm_middle.append('ry({{}}) q[{}];\n'.format(qubit))
            else:
                qasm_middle.append('rx({{}}) q[{}];\n'.format(qubit))

        qasm = ''.join(qasm_cnots_even) + ''.join(qasm_cnots_odd) + ''.join(qasm_middle) + ''.join(qasm_cnots_odd)\
               + ''.join(qasm_cnots_even)

        return qasm

    def get_ansatz_element(self, double_parameters=False):

        qasm = self.get_element_qasm(double_parameters)
        # return just a single ansatz element
        return AnsatzElement(element=qasm, element_type=self.ansatz_type,
                             n_var_parameters=(1+double_parameters)*self.n_orbitals)


class HeuristicAnsatz2:
    def __init__(self, n_orbitals, n_electrons):
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons
        self.ansatz_type = 'hardware_efficient'

    def get_element_qasm(self, double_parameters=False):

        qasm_singles = ['']
        qasm_cnots = ['']

        # apply single qubit general rotations to each qubit
        for qubit in range(self.n_orbitals):
            if qubit < self.n_orbitals-1:
                qasm_cnots.append('cx q[{}], q[{}];\n'.format(qubit, qubit + 1))

            if double_parameters:
                qasm_singles.append('rx({{}}) q[{}];\n'.format(qubit))  # we want to leave first {} empty for var_parameter later
                qasm_singles.append('ry({{}}) q[{}];\n'.format(qubit))
            else:
                qasm_singles.append('rx({{}}) q[{}];\n'.format(qubit))

        qasm_middle = 'rz({{}}) q[{}];\n'.format(self.n_orbitals - 1)

        qasm = ''.join(qasm_singles) + ''.join(qasm_cnots) + ''.join(qasm_middle) # + ''.join(qasm_cnots[::-1]) \
              # + ''.join(qasm_singles)

        return qasm

    def get_ansatz_element(self, double_parameters=False):

        qasm = self.get_element_qasm(double_parameters)
        # return just a single ansatz element
        return AnsatzElement(element=qasm, element_type=self.ansatz_type,
                             n_var_parameters=1*(1+double_parameters)*self.n_orbitals+1)


class HeuristicAnsatz3:
    def __init__(self, n_orbitals, n_electrons):
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons
        self.ansatz_type = 'hardware_efficient'
        self.n_parameters = 0

    def get_element_qasm(self, double_parameters=False, index=3):
        self.n_parameters = 0
        qasm = ['']
        for i in range(1, index + 1):
            used_qubits = numpy.zeros(self.n_orbitals)
            for qubit in range(self.n_orbitals):
                ctrl_qubit = qubit
                target_qubit = (qubit + index) % self.n_orbitals
                if used_qubits[target_qubit] == 0 or used_qubits[ctrl_qubit] == 0:
                    qasm.append('cx q[{}], q[{}];\n'.format(ctrl_qubit, target_qubit))

                    if double_parameters:
                        qasm.append('rx({{}}) q[{}];\n'.format(ctrl_qubit))
                        qasm.append('rx({{}}) q[{}];\n'.format(target_qubit))
                        qasm.append('ry({{}}) q[{}];\n'.format(ctrl_qubit))
                        qasm.append('ry({{}}) q[{}];\n'.format(target_qubit))
                        self.n_parameters += 4
                    else:
                        qasm.append('rx({{}}) q[{}];\n'.format(ctrl_qubit))
                        qasm.append('rx({{}}) q[{}];\n'.format(target_qubit))
                        self.n_parameters += 2

                    qasm.append('cx q[{}], q[{}];\n'.format(ctrl_qubit, target_qubit))

                    if double_parameters:
                        qasm.append('rx({{}}) q[{}];\n'.format(ctrl_qubit))
                        qasm.append('rx({{}}) q[{}];\n'.format(target_qubit))
                        qasm.append('ry({{}}) q[{}];\n'.format(ctrl_qubit))
                        qasm.append('ry({{}}) q[{}];\n'.format(target_qubit))
                        self.n_parameters += 4
                    else:
                        qasm.append('ry({{}}) q[{}];\n'.format(ctrl_qubit))
                        qasm.append('ry({{}}) q[{}];\n'.format(target_qubit))
                        self.n_parameters += 2

                    used_qubits[target_qubit] = 1
                    used_qubits[ctrl_qubit] = 1

        return ''.join(qasm)

    def get_ansatz_element(self, double_parameters=False, index=3):

        qasm = self.get_element_qasm(double_parameters, index)
        # return just a single ansatz element
        return AnsatzElement(element=qasm, element_type=self.ansatz_type,
                             n_var_parameters=self.n_parameters)