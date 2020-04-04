from openfermion import QubitOperator, FermionOperator
from openfermion.transforms import jordan_wigner

from src.utils import QasmUtils, MatrixUtils

import itertools
import numpy


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< individual ansatz elements >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class AnsatzElement:
    def __init__(self, element_type, element, excitation=None, n_var_parameters=1, excitation_order=None):
        self.excitation = excitation
        self.element_type = element_type
        self.n_var_parameters = n_var_parameters
        self.element = element

        if (self.element_type == 'excitation') and (excitation_order is None):
            assert type(self.excitation) == QubitOperator
            assert n_var_parameters == 1
            self.excitation_order = self.get_excitation_order()
        else:
            self.excitation_order = excitation_order

    def get_qasm(self, var_parameters):
        if self.element_type == 'excitation':
            assert len(var_parameters) == 1
            return QasmUtils.excitation_qasm(self.excitation, var_parameters[0])
        else:
            var_parameters = numpy.array(var_parameters)
            # TODO
            # return self.excitation.format(*var_parameters)
            return self.element.format(*var_parameters)

    def get_excitation_order(self):
        terms = list(self.excitation)
        n_terms = len(terms)
        return max([len(terms[i]) for i in range(n_terms)])


class SingleExcitation(AnsatzElement):
    def __init__(self, qubit_1, qubit_2):
        self.qubit_1 = qubit_1
        self.qubit_2 = qubit_2
        fermi_operator = FermionOperator('[{1}^ {0}] - [{0}^ {1}]'.format(qubit_2, qubit_1))
        excitation = jordan_wigner(fermi_operator)

        super(SingleExcitation, self).__init__(element=fermi_operator, excitation=excitation, excitation_order=1
                                               , element_type='excitation', n_var_parameters=1)


class DoubleExcitation(AnsatzElement):
    def __init__(self, qubit_pair_1, qubit_pair_2):
        self.qubit_pair_1 = qubit_pair_1
        self.qubit_pair_2 = qubit_pair_2

        fermi_operator = FermionOperator('[{2}^ {3}^ {0} {1}] - [{0}^ {1}^ {2} {3}]'
                                         .format(qubit_pair_1[0], qubit_pair_1[1], qubit_pair_2[0], qubit_pair_2[1]))
        excitation = jordan_wigner(fermi_operator)

        super(DoubleExcitation, self).__init__(element=fermi_operator, excitation=excitation, excitation_order=2
                                               , element_type='excitation', n_var_parameters=1)


class SingleExchange(AnsatzElement):
    def __init__(self, qubit_1, qubit_2):
        self.qubit_1 = qubit_1
        self.qubit_2 = qubit_2
        super(SingleExchange, self).__init__(element='s_exc {}, {}'.format(qubit_1, qubit_2)
                                             , element_type=str(self), n_var_parameters=1)

    def get_qasm(self, var_parameters):
        assert len(var_parameters) == 1
        return QasmUtils.partial_exchange_gate(var_parameters[0], self.qubit_1, self.qubit_2)


class DoubleExchange(AnsatzElement):
    def __init__(self, qubit_pair_1, qubit_pair_2, rescaled_parameter=False, parity_dependence=False, d_exc_correction=False):
        self.qubit_pair_1 = qubit_pair_1
        self.qubit_pair_2 = qubit_pair_2
        self.rescaled_parameter = rescaled_parameter
        self.parity_dependence = parity_dependence
        self.d_exc_correction = d_exc_correction
        super(DoubleExchange, self).__init__(element='d_exc {}, {}'.format(qubit_pair_1, qubit_pair_2),
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
        qasm = ['']
        qasm.append(QasmUtils.partial_exchange_gate(angle, qubit_pair_1[1], qubit_pair_2[0]))
        qasm.append(QasmUtils.partial_exchange_gate(-angle, qubit_pair_1[0], qubit_pair_2[1]))
        qasm.append('cz q[{}], q[{}];\n'.format(qubit_pair_2[0], qubit_pair_2[1]))

        # correction 3rd order terms approximates the operation of a double exchange
        if d_exc_correction:
            angle_2 = DoubleExchange.second_angle(angle)
        # not correcting 3rd order terms approximates the operation of a double excitation (with 3rd order error terms)
        else:
            angle_2 = angle
        qasm.append(QasmUtils.partial_exchange_gate(-angle_2, qubit_pair_1[1], qubit_pair_2[0]))
        qasm.append(QasmUtils.partial_exchange_gate(angle_2, qubit_pair_1[0], qubit_pair_2[1]))

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

        # this method constructs an operation that acts approximately as a double partial exchange

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

        return self.custom_double_excitation(parameter, self.qubit_pair_1, self.qubit_pair_2) #, parity_dependence=self.parity_dependence)


# <<<<<<<<<<<<<<<<<<<<<<<<<< ansatzes (lists of ansatz elements) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# exchange single and double
class ESD:
    def __init__(self, n_orbitals, n_electrons, rescaled=False, parity_dependence=False, d_exc_correction=False):
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons
        self.rescaled = rescaled
        self.parity_dependence = parity_dependence
        self.d_exc_correction = d_exc_correction

    def get_single_exchanges(self):
        single_excitations = []
        for i in range(self.n_electrons):
            for j in range(self.n_electrons, self.n_orbitals):
                single_excitations.append(SingleExchange(i, j))

        return single_excitations

    def get_double_exchanges(self):
        double_excitations = []
        for i in range(self.n_electrons - 1):
            for j in range(i + 1, self.n_electrons):
                for k in range(self.n_electrons, self.n_orbitals - 1):
                    for l in range(k + 1, self.n_orbitals):
                        double_excitations.append(DoubleExchange([i, j], [k, l], rescaled_parameter=self.rescaled,
                                                                 parity_dependence=self.parity_dependence,
                                                                 d_exc_correction=self.d_exc_correction))

        return double_excitations

    def get_ansatz_elements(self):
        return self.get_single_exchanges() + self.get_double_exchanges()


class EGSD:
    def __init__(self, n_orbitals, n_electrons):
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons

    def get_single_exchanges(self):
        single_excitations = []
        for indices in itertools.combinations(range(self.n_orbitals), 2):
            single_excitations.append(SingleExchange(*indices))

        return single_excitations

    def get_double_exchanges(self):
        double_excitations = []
        for indices in itertools.combinations(range(self.n_orbitals), 4):
            double_excitations.append(DoubleExchange(indices[:2], indices[-2:]))

        return double_excitations

    def get_ansatz_elements(self):
        return self.get_single_exchanges() + self.get_double_exchanges()


class UCCSD:
    def __init__(self, n_orbitals, n_electrons):
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons

    def get_single_excitation_list(self):
        single_excitations = []
        for i in range(self.n_electrons):
            for j in range(self.n_electrons, self.n_orbitals):
                # fermi_operator = FermionOperator('[{1}^ {0}] - [{0}^ {1}]'.format(j, i))
                # excitation = jordan_wigner(fermi_operator)
                # single_excitations.append(AnsatzElement('excitation', excitation=excitation, element=fermi_operator,
                #                                         excitation_order=1))
                single_excitations.append(SingleExcitation(i, j))
        return single_excitations

    def get_double_excitation_list(self):
        double_excitations = []
        for i in range(self.n_electrons-1):
            for j in range(i+1, self.n_electrons):
                for k in range(self.n_electrons, self.n_orbitals-1):
                    for l in range(k+1, self.n_orbitals):
                        # fermi_operator = FermionOperator('[{2}^ {3}^ {0} {1}] - [{0}^ {1}^ {2} {3}]'.format(i, j, k, l))
                        # excitation = jordan_wigner(fermi_operator)
                        # double_excitations.append(AnsatzElement('excitation', excitation=excitation, element=fermi_operator,
                        #                                         excitation_order=2))
                        double_excitations.append(DoubleExcitation([i, j], [k, l]))
        return double_excitations

    def get_ansatz_elements(self):
        return self.get_single_excitation_list() + self.get_double_excitation_list()


class UCCGSD:
    def __init__(self, n_orbitals, n_electrons):
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons

    def get_single_excitation_list(self):
        single_excitations = []
        for indices in itertools.combinations(range(self.n_orbitals), 2):
            fermi_operator = FermionOperator('[{1}^ {0}] - [{0}^ {1}]'.format(* indices))
            excitation = jordan_wigner(fermi_operator)
            single_excitations.append(AnsatzElement('excitation', excitation=excitation, element=fermi_operator,
                                                    excitation_order=1))
        return single_excitations

    def get_double_excitation_list(self):
        double_excitations = []
        for indices in itertools.combinations(range(self.n_orbitals), 4):
            fermi_operator = FermionOperator('[{2}^ {3}^ {0} {1}] - [{0}^ {1}^ {2} {3}]'.format(* indices))
            excitation = jordan_wigner(fermi_operator)
            double_excitations.append(AnsatzElement('excitation', excitation=excitation, element=fermi_operator,
                                                    excitation_order=2))
        return double_excitations

    def get_ansatz_elements(self):
        return self.get_single_excitation_list() + self.get_double_excitation_list()



