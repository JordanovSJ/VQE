from openfermion import QubitOperator, FermionOperator
from openfermion.transforms import jordan_wigner
from openfermion.utils import hermitian_conjugated

from src.utils import QasmUtils, MatrixUtils

import itertools
import numpy


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< individual ansatz elements >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class AnsatzElement:
    def __init__(self, element, n_var_parameters=1, order=None):
        self.order = order
        self.n_var_parameters = n_var_parameters
        self.element = element

    @staticmethod
    def get_qubit_excitation(qubits_1, qubits_2):

        assert len(qubits_2) == len(qubits_1)

        term_1 = QubitOperator('')

        for qubit_1, qubit_2 in zip(qubits_1, qubits_2):
            term_1 *= (QubitOperator('X{}'.format(qubit_1), 0.5) - QubitOperator('Y{}'.format(qubit_1), 0.5j))
            term_1 *= (QubitOperator('X{}'.format(qubit_2), 0.5) + QubitOperator('Y{}'.format(qubit_2), 0.5j))

        term_2 = hermitian_conjugated(term_1)

        return term_1 - term_2
    # # TODO not used
    # def get_qasm(self, var_parameters):
    #     if self.element_type == 'excitation':
    #         assert len(var_parameters) == 1
    #         return QasmUtils.fermi_excitation(self.excitation, var_parameters[0])
    #     else:
    #         var_parameters = numpy.array(var_parameters)
    #         # TODO
    #         # return self.excitation.format(*var_parameters)
    #         return self.element.format(*var_parameters)


class SingleFermiExcitation(AnsatzElement):
    def __init__(self, qubit_1, qubit_2):
        self.qubit_1 = qubit_1
        self.qubit_2 = qubit_2

        fermi_operator = FermionOperator('[{1}^ {0}] - [{0}^ {1}]'.format(self.qubit_2, self.qubit_1))
        self.excitation = jordan_wigner(fermi_operator)

        super(SingleFermiExcitation, self).\
            __init__(element='s_f_exc_{}_{}'.format(qubit_1, qubit_2), order=1, n_var_parameters=1)

    def get_qasm(self, var_parameters):
        assert len(var_parameters) == 1
        return QasmUtils.fermi_excitation(self.excitation, var_parameters[0])


class DoubleFermiExcitation(AnsatzElement):
    def __init__(self, qubit_pair_1, qubit_pair_2):
        self.qubit_pair_1 = qubit_pair_1
        self.qubit_pair_2 = qubit_pair_2

        fermi_operator = FermionOperator('[{2}^ {3}^ {0} {1}] - [{0}^ {1}^ {2} {3}]'
                                         .format(self.qubit_pair_1[0], self.qubit_pair_1[1],
                                                 self.qubit_pair_2[0], self.qubit_pair_2[1]))
        self.excitation = jordan_wigner(fermi_operator)

        super(DoubleFermiExcitation, self).\
            __init__(element='d_f_exc_{}_{}'.format(qubit_pair_1, qubit_pair_2), order=2, n_var_parameters=1)

    def get_qasm(self, var_parameters):
        assert len(var_parameters) == 1

        return QasmUtils.fermi_excitation(self.excitation, var_parameters[0])


class SingleQubitExcitation(AnsatzElement):
    def __init__(self, qubit_1, qubit_2):
        self.qubit_1 = qubit_1
        self.qubit_2 = qubit_2
        self.excitation = self.get_qubit_excitation([qubit_1], [qubit_2])
        super(SingleQubitExcitation, self).\
            __init__(element='s_q_exc_{}_{}'.format(qubit_1, qubit_2),  order=1, n_var_parameters=1)

    def get_qasm(self, var_parameters):
        assert len(var_parameters) == 1
        return QasmUtils.partial_exchange(var_parameters[0], self.qubit_1, self.qubit_2)


class DoubleQubitExcitation(AnsatzElement):
    def __init__(self, qubit_pair_1, qubit_pair_2):
        self.qubit_pair_1 = qubit_pair_1
        self.qubit_pair_2 = qubit_pair_2
        self.excitation = self.get_qubit_excitation(qubit_pair_1, qubit_pair_2)
        super(DoubleQubitExcitation, self).\
            __init__(element='d_q_exc_{}_{}'.format(qubit_pair_1, qubit_pair_2),  order=2, n_var_parameters=1)

    @staticmethod
    def double_qubit_excitation(angle, qubit_pair_1, qubit_pair_2):
        qasm = ['']
        angle = angle * 2  # for consistency with the conventional fermi excitation
        theta = angle / 8

        # determine the parity of the two pairs
        qasm.append('cx q[{}], q[{}];\n'.format(*qubit_pair_1))
        qasm.append('x q[{}];\n'.format(qubit_pair_1[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(*qubit_pair_2))
        qasm.append('x q[{}];\n'.format(qubit_pair_2[1]))

        # apply a partial swap of qubits 0 and 2, controlled by 1 and 3 ##

        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_2[0]))
        # # partial ccc_y operation
        qasm.append('rz({}) q[{}];\n'.format(numpy.pi / 2, qubit_pair_1[0]))

        qasm.append('rx({}) q[{}];\n'.format(theta, qubit_pair_1[0]))  # +

        qasm.append('h q[{}];\n'.format(qubit_pair_1[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_1[1]))  # 0 1
        qasm.append('h q[{}];\n'.format(qubit_pair_1[1]))

        qasm.append('rx({}) q[{}];\n'.format(-theta, qubit_pair_1[0]))  # -

        qasm.append('h q[{}];\n'.format(qubit_pair_2[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_2[1]))  # 0 3
        qasm.append('h q[{}];\n'.format(qubit_pair_2[1]))

        qasm.append('rx({}) q[{}];\n'.format(theta, qubit_pair_1[0]))  # +

        qasm.append('h q[{}];\n'.format(qubit_pair_1[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_1[1]))  # 0 1
        qasm.append('h q[{}];\n'.format(qubit_pair_1[1]))

        qasm.append('rx({}) q[{}];\n'.format(-theta, qubit_pair_1[0]))  # -

        qasm.append('h q[{}];\n'.format(qubit_pair_2[0]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_2[0]))  # 0 2
        qasm.append('h q[{}];\n'.format(qubit_pair_2[0]))

        qasm.append('rx({}) q[{}];\n'.format(theta, qubit_pair_1[0]))  # +

        qasm.append('h q[{}];\n'.format(qubit_pair_1[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_1[1]))  # 0 1
        qasm.append('h q[{}];\n'.format(qubit_pair_1[1]))

        qasm.append('rx({}) q[{}];\n'.format(-theta, qubit_pair_1[0]))  # -

        qasm.append('h q[{}];\n'.format(qubit_pair_2[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_2[1]))  # 0 3
        qasm.append('h q[{}];\n'.format(qubit_pair_2[1]))

        qasm.append('rx({}) q[{}];\n'.format(theta, qubit_pair_1[0]))  # +

        qasm.append('h q[{}];\n'.format(qubit_pair_1[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_1[1]))  # 0 1
        qasm.append('h q[{}];\n'.format(qubit_pair_1[1]))

        qasm.append('rx({}) q[{}];\n'.format(-theta, qubit_pair_1[0]))  # -

        qasm.append('rz({}) q[{}];\n'.format(-numpy.pi / 2, qubit_pair_1[0]))

        # ############################## partial ccc_y operation  ############ to here

        qasm.append(QasmUtils.controlled_xz(qubit_pair_1[0], qubit_pair_2[0], reverse=True))

        # correct for parity determination
        qasm.append('x q[{}];\n'.format(qubit_pair_1[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(*qubit_pair_1))
        qasm.append('x q[{}];\n'.format(qubit_pair_2[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(*qubit_pair_2))

        return ''.join(qasm)

    def get_qasm(self, var_parameters):
        assert len(var_parameters) == 1
        parameter = var_parameters[0]

        return self.double_qubit_excitation(parameter, self.qubit_pair_1, self.qubit_pair_2)


class EfficientSingleFermiExcitation(AnsatzElement):
    def __init__(self, qubit_1, qubit_2):
        self.qubit_1 = qubit_1
        self.qubit_2 = qubit_2

        fermi_operator = FermionOperator('[{1}^ {0}] - [{0}^ {1}]'.format(self.qubit_2, self.qubit_1))
        self.excitation = jordan_wigner(fermi_operator)

        super(EfficientSingleFermiExcitation, self).\
            __init__(element='eff_s_f_exc_{}_{}'.format(qubit_2, qubit_1), order=1, n_var_parameters=1)

    @staticmethod
    def efficient_single_fermi_excitation(angle, qubit_1, qubit_2):
        theta = numpy.pi / 2 + angle
        qasm = ['']
        if qubit_2 < qubit_1:
            x = qubit_1
            qubit_1 = qubit_2
            qubit_2 = x

        parity_qubits = list(range(qubit_1 + 1, qubit_2))

        parity_cnot_ladder = ['']
        if len(parity_qubits) > 0:
            for i in range(len(parity_qubits) - 1):
                parity_cnot_ladder.append('cx q[{}], q[{}];\n'.format(parity_qubits[i], parity_qubits[i + 1]))

            qasm += parity_cnot_ladder
            # parity dependence
            qasm.append('h q[{}];\n'.format(qubit_1))
            qasm.append('cx q[{}], q[{}];\n'.format(parity_qubits[-1], qubit_1))
            qasm.append('h q[{}];\n'.format(qubit_1))

        qasm.append(QasmUtils.controlled_xz(qubit_2, qubit_1))

        qasm.append('ry({}) q[{}];\n'.format(theta, qubit_2))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_1, qubit_2))
        qasm.append('ry({}) q[{}];\n'.format(-theta, qubit_2))

        qasm.append('cx q[{}], q[{}];\n'.format(qubit_2, qubit_1))

        if len(parity_qubits) > 0:
            qasm.append('h q[{}];\n'.format(qubit_1))
            qasm.append('cx q[{}], q[{}];\n'.format(parity_qubits[-1], qubit_1))
            qasm.append('h q[{}];\n'.format(qubit_1))

            qasm += parity_cnot_ladder[::-1]

        return ''.join(qasm)

    def get_qasm(self, var_parameters):
        assert len(var_parameters) == 1
        return self.efficient_single_fermi_excitation(var_parameters[0], self.qubit_1, self.qubit_2)


class EfficientDoubleFermiExcitation(AnsatzElement):
    def __init__(self, qubit_pair_1, qubit_pair_2):
        self.qubit_pair_1 = qubit_pair_1
        self.qubit_pair_2 = qubit_pair_2

        fermi_operator = FermionOperator('[{2}^ {3}^ {0} {1}] - [{0}^ {1}^ {2} {3}]'
                                         .format(self.qubit_pair_1[0], self.qubit_pair_1[1],
                                                 self.qubit_pair_2[0], self.qubit_pair_2[1]))
        self.excitation = jordan_wigner(fermi_operator)

        super(EfficientDoubleFermiExcitation, self).\
            __init__(element='eff_d_f_exc_{}_{}'.format(qubit_pair_1, qubit_pair_2),  order=2, n_var_parameters=1)

    @staticmethod
    def efficient_double_fermi_excitation(angle, qubit_pair_1, qubit_pair_2):
        angle = - angle * 2  # the factor of -2 is for consistency with the conventional fermi excitation
        theta = angle / 8

        qasm = ['']
        qubit_pair_1.sort()
        qubit_pair_2.sort()

        # do not include the first qubits of qubit_pair_1 and qubit_pair_2
        parity_qubits = list(range(qubit_pair_1[0] + 1, qubit_pair_1[1])) + list(range(qubit_pair_2[0] + 1, qubit_pair_2[1]))

        # ladder of CNOT used to determine the parity
        parity_cnot_ladder = ['']
        if len(parity_qubits) > 0:
            for i in range(len(parity_qubits) - 1):
                parity_cnot_ladder.append('cx q[{}], q[{}];\n'.format(parity_qubits[i], parity_qubits[i + 1]))
            # parity_cnot_ladder.append('x q[{}];\n'.format(parity_qubits[-1]))

        # determine the parity of the two pairs
        qasm.append('cx q[{}], q[{}];\n'.format(*qubit_pair_1))
        qasm.append('x q[{}];\n'.format(qubit_pair_1[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(*qubit_pair_2))
        qasm.append('x q[{}];\n'.format(qubit_pair_2[1]))

        # apply a partial swap of qubits 0 and 2, controlled by 1 and 3 ##

        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_2[0]))

        # apply parity sign correction 1
        if len(parity_qubits) > 0:
            qasm += parity_cnot_ladder
            qasm.append('h q[{}];\n'.format(parity_qubits[-1]))
            qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], parity_qubits[-1]))

        # # partial ccc_y operation
        qasm.append('rz({}) q[{}];\n'.format(numpy.pi / 2, qubit_pair_1[0]))

        qasm.append('rx({}) q[{}];\n'.format(theta, qubit_pair_1[0]))  # +

        qasm.append('h q[{}];\n'.format(qubit_pair_1[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_1[1]))  # 0 1
        qasm.append('h q[{}];\n'.format(qubit_pair_1[1]))

        qasm.append('rx({}) q[{}];\n'.format(-theta, qubit_pair_1[0]))  # -

        qasm.append('h q[{}];\n'.format(qubit_pair_2[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_2[1]))  # 0 3
        qasm.append('h q[{}];\n'.format(qubit_pair_2[1]))

        qasm.append('rx({}) q[{}];\n'.format(theta, qubit_pair_1[0]))  # +

        qasm.append('h q[{}];\n'.format(qubit_pair_1[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_1[1]))  # 0 1
        qasm.append('h q[{}];\n'.format(qubit_pair_1[1]))

        qasm.append('rx({}) q[{}];\n'.format(-theta, qubit_pair_1[0]))  # -

        qasm.append('h q[{}];\n'.format(qubit_pair_2[0]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_2[0]))  # 0 2
        qasm.append('h q[{}];\n'.format(qubit_pair_2[0]))

        qasm.append('rx({}) q[{}];\n'.format(theta, qubit_pair_1[0]))  # +

        qasm.append('h q[{}];\n'.format(qubit_pair_1[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_1[1]))  # 0 1
        qasm.append('h q[{}];\n'.format(qubit_pair_1[1]))

        qasm.append('rx({}) q[{}];\n'.format(-theta, qubit_pair_1[0]))  # -

        qasm.append('h q[{}];\n'.format(qubit_pair_2[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_2[1]))  # 0 3
        qasm.append('h q[{}];\n'.format(qubit_pair_2[1]))

        qasm.append('rx({}) q[{}];\n'.format(theta, qubit_pair_1[0]))  # +

        qasm.append('h q[{}];\n'.format(qubit_pair_1[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_1[1]))  # 0 1
        qasm.append('h q[{}];\n'.format(qubit_pair_1[1]))

        qasm.append('rx({}) q[{}];\n'.format(-theta, qubit_pair_1[0]))  # -

        qasm.append('rz({}) q[{}];\n'.format(-numpy.pi / 2, qubit_pair_1[0]))

        # ############################## partial ccc_y operation  ############ to here

        # apply parity sign correction 2
        if len(parity_qubits) > 0:
            qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], parity_qubits[-1]))
            qasm.append('h q[{}];\n'.format(parity_qubits[-1]))
            qasm += parity_cnot_ladder[::-1]

        qasm.append(QasmUtils.controlled_xz(qubit_pair_1[0], qubit_pair_2[0], reverse=True))

        # correct for parity determination
        qasm.append('x q[{}];\n'.format(qubit_pair_1[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(*qubit_pair_1))
        qasm.append('x q[{}];\n'.format(qubit_pair_2[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(*qubit_pair_2))

        return ''.join(qasm)

    def get_qasm(self, var_parameters):
        assert len(var_parameters) == 1
        parameter = var_parameters[0]

        return self.efficient_double_fermi_excitation(parameter, self.qubit_pair_1, self.qubit_pair_2)


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

        qasm.append('cx q[0], q[1];\n'.format(qubit_1, qubit_2))
        qasm.append('rx({}) q[{}];\n'.format(angle, qubit_1))
        qasm.append('rz({}) q[{}];\n'.format(angle, qubit_2))
        qasm.append('cx q[0], q[1];\n'.format(qubit_1, qubit_2))

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


# Not used :(
class DoubleExchange(AnsatzElement):
    def __init__(self, qubit_pair_1, qubit_pair_2, rescaled_parameter=False, parity_dependence=False, d_exc_correction=False):
        self.qubit_pair_1 = qubit_pair_1
        self.qubit_pair_2 = qubit_pair_2
        self.rescaled_parameter = rescaled_parameter
        self.parity_dependence = parity_dependence
        self.d_exc_correction = d_exc_correction
        super(DoubleExchange, self).\
            __init__(element='d_exchange_{}_{}'.format(qubit_pair_1, qubit_pair_2),  order=2, n_var_parameters=1)

    @staticmethod
    def second_angle(x):
        if x == 0:
            return 0
        else:
            tan_x = numpy.tan(x)
            tan_x_squared = tan_x**2
            tan_y = ((-tan_x_squared - 1 + numpy.sqrt(tan_x_squared ** 2 + 6 * tan_x_squared + 1)) / (2*tan_x))
            return numpy.arctan(tan_y)

    # this method constructs an operation that acts approximately as a double partial exchange
    @staticmethod
    def double_exchange(angle, qubit_pair_1, qubit_pair_2, parity_dependence=False, d_exc_correction=False):
        assert len(qubit_pair_1) == 2
        assert len(qubit_pair_2) == 2

        theta_1 = numpy.pi / 2 - angle

        qasm = ['']

        # 1st exchange + 0-2
        qasm.append(QasmUtils.controlled_xz(qubit_pair_2[0], qubit_pair_1[0]))
        qasm.append('ry({}) q[{}];\n'.format(theta_1, qubit_pair_2[0]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_2[0]))
        qasm.append('ry({}) q[{}];\n'.format(-theta_1, qubit_pair_2[0]))

        # 2nd exchange + 1-3
        qasm.append(QasmUtils.controlled_xz(qubit_pair_2[1], qubit_pair_1[1]))
        qasm.append('ry({}) q[{}];\n'.format(theta_1, qubit_pair_2[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[1], qubit_pair_2[1]))
        qasm.append('ry({}) q[{}];\n'.format(-theta_1, qubit_pair_2[1]))

        # CZ gates
        qasm.append('cz q[{}], q[{}];\n'.format(qubit_pair_2[0], qubit_pair_2[1]))

        # correction 3rd order terms approximates the operation of a double exchange
        if d_exc_correction:
            angle_2 = DoubleExchange.second_angle(angle)
        # not correcting 3rd order terms approximates the operation of a double excitation (with 3rd order error terms)
        else:
            angle_2 = angle
        theta_2 = numpy.pi / 2 - angle_2

        # 3rd exchange - 0-2
        qasm.append('ry({}) q[{}];\n'.format(theta_2, qubit_pair_2[0]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_2[0]))
        qasm.append('ry({}) q[{}];\n'.format(-theta_2, qubit_pair_2[0]))
        qasm.append(QasmUtils.controlled_xz(qubit_pair_2[0], qubit_pair_1[0], reverse=True))

        # 4th exchange -1-3
        qasm.append('ry({}) q[{}];\n'.format(theta_2, qubit_pair_2[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[1], qubit_pair_2[1]))
        qasm.append('ry({}) q[{}];\n'.format(-theta_2, qubit_pair_2[1]))
        qasm.append(QasmUtils.controlled_xz(qubit_pair_2[1], qubit_pair_1[1], reverse=True))

        if parity_dependence:
            # do not include the first qubit of the second pair
            parity_qubits = list(range(min(qubit_pair_1), max(qubit_pair_1))) + list(range(min(qubit_pair_2)+1, max(qubit_pair_2)))

            # ladder of CNOT used to determine the parity
            cnot_ladder = ['']
            for i in range(len(parity_qubits) - 1):
                cnot_ladder.append('cx q[{}], q[{}];\n'.format(parity_qubits[i], parity_qubits[i+1]))

            if angle > 0:
                # applies a CZ correction in front, to get a negative sign for the excitation term, if the parity is 1
                # (or the parity of "parity_qubits" is 0)
                front = ['']
                # this is the CZ that determines the sign of the excitation term
                front.append('cz q[{}], q[{}];\n'.format(qubit_pair_2[0], qubit_pair_2[1]))
                # this bit determines the parity and applies a  CZ to negate the correction if the parity is wrong
                front += cnot_ladder
                front.append('x q[{}];\n'.format(parity_qubits[-1]))
                front.append('cz q[{}], q[{}];\n'.format(parity_qubits[-1], qubit_pair_2[0]))
                front.append('x q[{}];\n'.format(parity_qubits[-1]))
                front += cnot_ladder[::-1]

                # .. positive sign for the excitation term, if the parity is 0 (or the parity of "parity_qubits" is 1)
                rear = ['']
                # .. sign correction
                rear.append('cz q[{}], q[{}];\n'.format(qubit_pair_2[0], qubit_pair_2[1]))
                # .. parity correction
                rear += cnot_ladder
                rear.append('cz q[{}], q[{}];\n'.format(parity_qubits[-1], qubit_pair_2[0]))
                rear += cnot_ladder[::-1]
                # additional correction of states 010 and 110
                rear.append('x q[{}];\n'.format(qubit_pair_2[1]))
                rear.append('cz q[{}], q[{}];\n'.format(qubit_pair_2[0], qubit_pair_2[1]))
                rear.append('x q[{}];\n'.format(qubit_pair_2[1]))

                qasm = front + qasm + rear
            else:
                front = ['']
                # sign correction
                front.append('cz q[{}], q[{}];\n'.format(qubit_pair_2[0], qubit_pair_2[1]))
                # parity correction
                front += cnot_ladder
                front.append('cz q[{}], q[{}];\n'.format(parity_qubits[-1], qubit_pair_2[0]))
                front += cnot_ladder[::-1]

                rear = ['']
                # sign correction
                rear.append('cz q[{}], q[{}];\n'.format(qubit_pair_2[0], qubit_pair_2[1]))
                # parity correction
                rear += cnot_ladder
                rear.append('x q[{}];\n'.format(parity_qubits[-1]))
                rear.append('cz q[{}], q[{}];\n'.format(parity_qubits[-1], qubit_pair_2[0]))
                rear.append('x q[{}];\n'.format(parity_qubits[-1]))
                rear += cnot_ladder[::-1]
                # 010 and 011 correction
                rear.append('x q[{}];\n'.format(qubit_pair_2[1]))
                rear.append('cz q[{}], q[{}];\n'.format(qubit_pair_2[0], qubit_pair_2[1]))
                rear.append('x q[{}];\n'.format(qubit_pair_2[1]))

                qasm = front + qasm + rear
        else:
            if angle > 0:
                # adding a correcting CZ gate at the end will result in a minus sign
                qasm.append('cz q[{}], q[{}];\n'.format(qubit_pair_2[0], qubit_pair_2[1]))
            else:
                # adding a correcting CZ gate at the front will result in a plus sign
                qasm = ['cz q[{}], q[{}];\n'.format(qubit_pair_2[0], qubit_pair_2[1])] + qasm

        return ''.join(qasm)

    def get_qasm(self, var_parameters):
        assert len(var_parameters) == 1
        parameter = var_parameters[0]

        # rescaled parameter (used for easier gradient optimization)
        if self.rescaled_parameter:
            if var_parameters[0] > 0:
                parameter = var_parameters[0] + numpy.tanh(var_parameters[0]**0.5)
            else:
                parameter = var_parameters[0] + numpy.tanh(-(-var_parameters[0])**0.5)

        return self.double_exchange(parameter, self.qubit_pair_1, self.qubit_pair_2,
                                    parity_dependence=self.parity_dependence, d_exc_correction=self.d_exc_correction)




