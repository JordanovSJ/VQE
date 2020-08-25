# this module contains functions not designated a better place
import scipy
import numpy
import itertools

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
            excitation_matrix_list.append(self.get_excitation_matrix(params[i] * excitation))

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
        return AnsatzElement(excitation=qasm, element_type=self.ansatz_type,
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
        return AnsatzElement(excitation=qasm, element_type=self.ansatz_type,
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
        return AnsatzElement(excitation=qasm, element_type=self.ansatz_type,
                             n_var_parameters=self.n_parameters)


class ExchangeAnsatz1(AnsatzElement):
    def __init__(self, n_orbitals, n_electrons, n_blocks=1):
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons
        self.n_blocks = n_blocks

        n_var_parameters = min(n_electrons, n_orbitals - n_electrons)*(1 + n_blocks)
        super(ExchangeAnsatz1, self).\
            __init__(excitation=None, element_type=str(self), n_var_parameters=n_var_parameters)

    def get_qasm(self, var_parameters):
        assert len(var_parameters) == self.n_var_parameters
        var_parameters_cycle = itertools.cycle(var_parameters)
        qasm = ['']
        for block in range(self.n_blocks):
            unoccupied_orbitals = list(range(self.n_electrons, self.n_orbitals))
            for occupied_orbital in reversed(range(0, self.n_electrons)):
                if len(unoccupied_orbitals) == 0:
                    break
                if occupied_orbital == self.n_electrons - 1:
                    virtual_orbital = self.n_electrons + block
                else:
                    virtual_orbital = min(unoccupied_orbitals)
                unoccupied_orbitals.remove(virtual_orbital)

                # add a phase rotation for the excited orbitals only
                angle = var_parameters_cycle.__next__()
                qasm.append('rz({}) q[{}];\n'.format(angle, virtual_orbital))

                angle = var_parameters_cycle.__next__()
                qasm.append(QasmUtils.partial_exchange(angle, occupied_orbital, virtual_orbital))

            # TODO add exchanges between the last unoccupied orbitals?

        return ''.join(qasm)

    @staticmethod
    def double_exchange_old(angle, qubit_pair_1, qubit_pair_2):
        assert len(qubit_pair_1) == 2
        assert len(qubit_pair_2) == 2
        qasm = ['']
        qasm.append(QasmUtils.partial_exchange(angle, qubit_pair_1[1], qubit_pair_2[0]))
        qasm.append(QasmUtils.partial_exchange(-angle, qubit_pair_1[0], qubit_pair_2[1]))
        qasm.append('cz q[{}], q[{}];\n'.format(qubit_pair_2[0], qubit_pair_2[1]))
        qasm.append(QasmUtils.partial_exchange(-angle, qubit_pair_1[1], qubit_pair_2[0]))
        qasm.append(QasmUtils.partial_exchange(angle, qubit_pair_1[0], qubit_pair_2[1]))
        # corrections
        qasm.append('cz q[{}], q[{}];\n'.format(qubit_pair_2[0], qubit_pair_2[1]))
        return ''.join(qasm)


def rescaling_for_double_exchange(parameter):
    if parameter > 0:
        rescaled_parameter = parameter + numpy.tanh(parameter ** 0.5)
    else:
        rescaled_parameter = parameter + numpy.tanh(-(-parameter) ** 0.5)

    return rescaled_parameter


class ExchangeAnsatzBlock(AnsatzElement):
    def __init__(self, n_orbitals, n_electrons):
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons
        n_var_parameters = int(n_orbitals/4) + n_orbitals

        super(ExchangeAnsatzBlock, self).\
            __init__(element_type=str(self), element='block', n_var_parameters=n_var_parameters)

    def get_qasm(self, var_parameters):
        var_parameters_cycle = itertools.cycle(var_parameters)
        count = 0
        qasm = ['']
        # add single qubit n rotations
        # for qubit in range(self.n_orbitals):
        #     qasm.append('rz ({}) q[{}];\n'.format(var_parameters_cycle.__next__(), qubit))
        #     count += 1
        # double orbital exchanges
        for qubit in range(self.n_orbitals):
            if qubit % 4 == 0:
                q_0 = qubit
                q_1 = (qubit + 1) % self.n_orbitals
                q_2 = (qubit + 2) % self.n_orbitals
                q_3 = (qubit + 3) % self.n_orbitals
                q_4 = (qubit + 4) % self.n_orbitals
                # qasm.append(DoubleExchangeAnsatzElement.double_exchange(var_parameters_cycle.__next__(), [q_0, q_1], [q_2, q_3]))
                # count += 1
                qasm.append(DoubleExchangeAnsatzElement.double_exchange(var_parameters_cycle.__next__(), [q_1, q_2], [q_3, q_4]))
                count += 1
        # single orbital exchanges
        for qubit in range(self.n_orbitals):
            if qubit % 2 == 0:
                qasm.append(QasmUtils.partial_exchange(var_parameters_cycle.__next__(),
                                                       qubit, (qubit + 1) % self.n_orbitals))
                count += 1
                qasm.append(QasmUtils.partial_exchange(var_parameters_cycle.__next__(),
                                                       (qubit+1) % self.n_orbitals, (qubit + 2) % self.n_orbitals))
                count += 1
        assert count == len(var_parameters)
        return ''.join(qasm)


# DOES NOT WORK!!!!!!!!!!!
# performs a CP operation on q1 and q2 if the parity of {q} is odd, where q1 belongs to {q}, and q2 does not
@staticmethod
def parity_controlled_phase_gate(parity_qubits, qubit_1, qubit_2, reverse=False):
    qasm = ['']
    # ladder of cnots
    cnots = ['']

    parity_qubits = list(parity_qubits)
    control_qubits = parity_qubits + [qubit_1]

    for i in range(len(control_qubits) - 1):
        cnots.append('cx q[{}], q[{}];\n'.format(control_qubits[i], control_qubits[i+1]))

    qasm += cnots
    if reverse:
        qasm.append('x q[{}];\n'.format(qubit_1))
        qasm.append('cz q[{}], q[{}];\n'.format(qubit_1, qubit_2))
        qasm.append('x q[{}];\n'.format(qubit_1))
    else:
        qasm.append('cz q[{}], q[{}];\n'.format(qubit_1, qubit_2))
    qasm += cnots[::-1]

    return ''.join(qasm)


class CustomDoubleExcitation(AnsatzElement):
    def __init__(self, qubit_pair_1, qubit_pair_2, parity_dependence=False):
        self.qubit_pair_1 = qubit_pair_1
        self.qubit_pair_2 = qubit_pair_2
        self.parity_dependence = parity_dependence
        super(CustomDoubleExcitation, self).__init__(element='d_exc {}, {}'.format(qubit_pair_1, qubit_pair_2),
                                                     element_type=str(self), n_var_parameters=1)

    @staticmethod
    def custom_double_excitation(angle, qubit_pair_1, qubit_pair_2):
        qasm = ['']

        # determine tha parity of the two qubit pairs
        qasm.append('cx q[{}], q[{}];\n'.format(*qubit_pair_1))
        qasm.append('cx q[{}], q[{}];\n'.format(*qubit_pair_2))

        # perform partial single qubit exchange on q0 and q2, controlled by q1 = |0> and q3 = |0>

        qasm.append('x q[{}];\n'.format(qubit_pair_1[1]))
        qasm.append('x q[{}];\n'.format(qubit_pair_2[1]))

        # TODO add parity dependence
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_2[0], qubit_pair_1[0]))
        qasm.append(QasmUtils.n_controlled_y_rotation(angle=angle, controls=[*qubit_pair_1, qubit_pair_2[1]],
                                                      target=qubit_pair_2[0]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_2[0], qubit_pair_1[0]))

        qasm.append('x q[{}];\n'.format(qubit_pair_1[1]))
        qasm.append('x q[{}];\n'.format(qubit_pair_2[1]))

        qasm.append('cx q[{}], q[{}];\n'.format(*qubit_pair_1))
        qasm.append('cx q[{}], q[{}];\n'.format(*qubit_pair_2))

        return ''.join(qasm)

    def get_qasm(self, var_parameters):
        assert len(var_parameters) == 1
        parameter = var_parameters[0]

        return self.custom_double_excitation(parameter, self.qubit_pair_1,
                                             self.qubit_pair_2)  # , parity_dependence=self.parity_dependence)


def optimized_d_bosonic_excitation(angle, qubit_pair_1, qubit_pair_2):
    qasm = ['']

    # 1st term
    qasm.append('rz({}) q[{}];\n'.format(-numpy.pi / 2, qubit_pair_2[1]))  # S^

    qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_2[1], qubit_pair_2[0]))
    qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_2[1], qubit_pair_1[1]))
    qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_2[1], qubit_pair_1[0]))

    qasm.append('h q[{}];\n'.format(qubit_pair_2[1]))

    qasm.append('rz({}) q[{}];\n'.format(angle, qubit_pair_2[1]))

    qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_2[0], qubit_pair_2[1]))

    # 2nd term
    qasm.append('rz({}) q[{}];\n'.format(angle, qubit_pair_2[1]))

    qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_2[1]))

    # 3rd term (-)
    qasm.append('rz({}) q[{}];\n'.format(-angle, qubit_pair_2[1]))

    qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_2[0], qubit_pair_2[1]))

    # 4rd term (-)
    qasm.append('rz({}) q[{}];\n'.format(-angle, qubit_pair_2[1]))
    qasm.append('rz({}) q[{}];\n'.format(-numpy.pi / 2, qubit_pair_1[1]))  # S^

    qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[1], qubit_pair_2[1]))

    # 5th term
    qasm.append('rz({}) q[{}];\n'.format(angle, qubit_pair_2[1]))

    qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_2[0], qubit_pair_2[1]))

    # 6th term
    qasm.append('rz({}) q[{}];\n'.format(angle, qubit_pair_2[1]))

    qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_2[1]))

    # 7th term
    qasm.append('rz({}) q[{}];\n'.format(-angle, qubit_pair_2[1]))

    qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_2[0], qubit_pair_2[1]))

    # 8th term
    qasm.append('rz({}) q[{}];\n'.format(-angle, qubit_pair_2[1]))
    qasm.append('h q[{}];\n'.format(qubit_pair_2[1]))

    qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_2[1], qubit_pair_2[0]))
    qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_2[1], qubit_pair_1[1]))
    qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_2[1], qubit_pair_1[0]))

    qasm.append('rz({}) q[{}];\n'.format(numpy.pi / 2, qubit_pair_1[1]))  # S

    return ''.join(qasm)


class ExpSingleQubitExcitation(AnsatzElement):
    def __init__(self, qubit_1, qubit_2):
        self.qubit_1 = qubit_1
        self.qubit_2 = qubit_2
        super(ExpSingleQubitExcitation, self).\
            __init__(element='s_q_exc_{}_{}'.format(qubit_1, qubit_2),  order=1, n_var_parameters=1)

    @staticmethod
    def exp_single_qubit_excitations(angle, qubit_1, qubit_2):
        qasm = ['']

        qasm.append('rz({}) q[{}];\n'.format(numpy.pi / 2, qubit_1))

        qasm.append('rx({}) q[{}];\n'.format(numpy.pi / 2, qubit_1))
        qasm.append('rx({}) q[{}];\n'.format(numpy.pi / 2, qubit_2))

        qasm.append('cx q[{}], q[{}];\n'.format(qubit_1, qubit_2))
        qasm.append('rx({}) q[{}];\n'.format(angle, qubit_1))
        qasm.append('rz({}) q[{}];\n'.format(angle, qubit_2))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_1, qubit_2))

        qasm.append('rx({}) q[{}];\n'.format(-numpy.pi / 2, qubit_1))
        qasm.append('rx({}) q[{}];\n'.format(-numpy.pi / 2, qubit_2))

        qasm.append('rz({}) q[{}];\n'.format(-numpy.pi / 2, qubit_1))

        return ''.join(qasm)

    def get_qasm(self, var_parameters):
        assert len(var_parameters) == 1
        return self.exp_single_qubit_excitations(var_parameters[0], self.qubit_1, self.qubit_2)


class ExpDoubleQubitExcitation(AnsatzElement):
    def __init__(self, qubit_pair_1, qubit_pair_2):
        self.qubit_pair_1 = qubit_pair_1
        self.qubit_pair_2 = qubit_pair_2
        super(ExpDoubleQubitExcitation, self).\
            __init__(element='d_q_exc_{}_{}'.format(qubit_pair_1, qubit_pair_2),  order=2, n_var_parameters=1)

    @staticmethod
    def exp_double_qubit_excitation(angle, qubit_pair_1, qubit_pair_2):
        qasm = ['']
        angle = angle * 2  # for consistency with the conventional fermi excitation
        theta = angle / 8

        # determine the parity of the two pairs
        qasm.append('cx q[{}], q[{}];\n'.format(*qubit_pair_1))
        qasm.append('x q[{}];\n'.format(qubit_pair_1[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(*qubit_pair_2))
        qasm.append('x q[{}];\n'.format(qubit_pair_2[1]))

        # apply a partial swap of qubits 0 and 2, controlled by 1 and 3 ##

        # exp exchange interaction  >>>>>>>>>>>>>>>>>>>>>>
        qasm.append('rz({}) q[{}];\n'.format(numpy.pi / 2, qubit_pair_1[0]))

        qasm.append('rx({}) q[{}];\n'.format(numpy.pi / 2, qubit_pair_1[0]))
        qasm.append('rx({}) q[{}];\n'.format(numpy.pi / 2, qubit_pair_2[0]))

        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_2[0]))

        # ccRX
        qasm.append('rx({}) q[{}];\n'.format(angle/4, qubit_pair_1[0]))
        qasm.append('cz q[{}], q[{}];\n'.format(qubit_pair_1[1], qubit_pair_1[0]))
        qasm.append('rx({}) q[{}];\n'.format(-angle / 4, qubit_pair_1[0]))
        qasm.append('cz q[{}], q[{}];\n'.format(qubit_pair_2[1], qubit_pair_1[0]))
        qasm.append('rx({}) q[{}];\n'.format(angle / 4, qubit_pair_1[0]))
        qasm.append('cz q[{}], q[{}];\n'.format(qubit_pair_1[1], qubit_pair_1[0]))
        qasm.append('rx({}) q[{}];\n'.format(-angle / 4, qubit_pair_1[0]))
        qasm.append('cz q[{}], q[{}];\n'.format(qubit_pair_2[1], qubit_pair_1[0]))

        # and ccRZ
        qasm.append('rz({}) q[{}];\n'.format(angle/4, qubit_pair_2[0]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_2[1], qubit_pair_2[0]))
        qasm.append('rz({}) q[{}];\n'.format(-angle/4, qubit_pair_2[0]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[1], qubit_pair_2[0]))
        qasm.append('rz({}) q[{}];\n'.format(angle/4, qubit_pair_2[0]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_2[1], qubit_pair_2[0]))
        qasm.append('rz({}) q[{}];\n'.format(-angle/4, qubit_pair_2[0]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[1], qubit_pair_2[0]))

        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_2[0]))

        qasm.append('rx({}) q[{}];\n'.format(-numpy.pi / 2, qubit_pair_1[0]))
        qasm.append('rx({}) q[{}];\n'.format(-numpy.pi / 2, qubit_pair_2[0]))

        qasm.append('rz({}) q[{}];\n'.format(-numpy.pi / 2, qubit_pair_1[0]))
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # correct for parity determination
        qasm.append('x q[{}];\n'.format(qubit_pair_1[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(*qubit_pair_1))
        qasm.append('x q[{}];\n'.format(qubit_pair_2[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(*qubit_pair_2))

        return ''.join(qasm)

    def get_qasm(self, var_parameters):
        assert len(var_parameters) == 1
        parameter = var_parameters[0]

        return self.exp_double_qubit_excitation(parameter, self.qubit_pair_1, self.qubit_pair_2)

