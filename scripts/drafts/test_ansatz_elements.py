from openfermion import QubitOperator, FermionOperator
from openfermion.transforms import jordan_wigner

from src.utils import QasmUtils, MatrixUtils
from src.ansatz_elements import AnsatzElement, DoubleExchange

import itertools
import numpy


class EfficientDoubleExchange(AnsatzElement):
    def __init__(self, qubit_pair_1, qubit_pair_2, rescaled_parameter=False, parity_dependence=False, d_exc_correction=False):
        self.qubit_pair_1 = qubit_pair_1
        self.qubit_pair_2 = qubit_pair_2
        self.rescaled_parameter = rescaled_parameter
        self.parity_dependence = parity_dependence
        self.d_exc_correction = d_exc_correction
        super(EfficientDoubleExchange, self).__init__(element='d_exc {}, {}'.format(qubit_pair_1, qubit_pair_2),
                                                      element_type=str(self), n_var_parameters=1)

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
        theta_1 = numpy.pi/2 - angle

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
        # qasm.append('cz q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_1[1]))

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

        # correcting for parameter sign
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
                # qasm.append('cz q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_1[1]))

            else:
                # adding a correcting CZ gate at the front will result in a plus sign
                qasm = ['cz q[{}], q[{}];\n'.format(qubit_pair_2[0], qubit_pair_2[1]),
                        ] \
                       + qasm
                # 'cz q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_1[1])
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


class EfficientDoubleExcitation2(AnsatzElement):
    def __init__(self, qubit_pair_1, qubit_pair_2):
        self.qubit_pair_1 = qubit_pair_1
        self.qubit_pair_2 = qubit_pair_2
        super(EfficientDoubleExcitation2, self).__init__(element='optimized_d_exc {}, {}'.format(qubit_pair_1, qubit_pair_2),
                                                         element_type=str(self), n_var_parameters=1)

    @staticmethod
    def efficient_double_excitation_2(angle, qubit_pair_1, qubit_pair_2):
        qasm = ['']
        theta = angle / 8

        # determine the parity of the two pairs
        qasm.append('cx q[{}], q[{}];\n'.format(*qubit_pair_1))
        qasm.append('x q[{}];\n'.format(qubit_pair_1[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(*qubit_pair_2))
        qasm.append('x q[{}];\n'.format(qubit_pair_2[1]))

        # apply a partial swap of qubits 0 and 2, controlled by 1 and 3 ##

        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_2[0]))
        # # partial ccc_y operation
        qasm.append('rz({}) q[{}];\n'.format(numpy.pi/2, qubit_pair_1[0]))

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

        ##### test #####
        # qasm.append('h q[{}];\n'.format(qubit_pair_2[0]))
        # qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_2[0]))  # 0 2
        # qasm.append('h q[{}];\n'.format(qubit_pair_2[0]))
        # ###############

        # partial ccc_y operation  ############ to here

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

        return self.efficient_double_excitation_2(parameter, self.qubit_pair_1, self.qubit_pair_2)

