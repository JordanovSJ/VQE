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
            return self.excitation.format(*var_parameters)

    def get_excitation_order(self):
        terms = list(self.excitation)
        n_terms = len(terms)
        return max([len(terms[i]) for i in range(n_terms)])


class ExchangeAnsatzElement(AnsatzElement):
    def __init__(self, qubit_1, qubit_2):
        self.qubit_1 = qubit_1
        self.qubit_2 = qubit_2
        super(ExchangeAnsatzElement, self).__init__(element='s_exc {}, {}'.format(qubit_1, qubit_2)
                                                    , element_type=str(self),  n_var_parameters=1)

    def get_qasm(self, var_parameters):
        assert len(var_parameters) == 1
        return QasmUtils.partial_exchange_gate_qasm(var_parameters[0], self.qubit_1, self.qubit_2)


# Heuristic exchange ansatz 1,  17.02.2020
class DoubleExchangeAnsatzElement(AnsatzElement):
    def __init__(self, qubit_pair_1, qubit_pair_2):
        self.qubit_pair_1 = qubit_pair_1
        self.qubit_pair_2 = qubit_pair_2
        super(DoubleExchangeAnsatzElement, self).__init__(element='d_exc {}, {}'.format(qubit_pair_1, qubit_pair_2),
                                                          element_type=str(self),  n_var_parameters=1)

    @staticmethod
    def double_exchange(angle, qubit_pair_1, qubit_pair_2):
        assert len(qubit_pair_1) == 2
        assert len(qubit_pair_2) == 2
        qasm = ['']
        qasm.append(QasmUtils.partial_exchange_gate_qasm(angle, qubit_pair_1[1], qubit_pair_2[0]))
        qasm.append(QasmUtils.partial_exchange_gate_qasm(angle, qubit_pair_1[0], qubit_pair_2[1]))
        qasm.append('cz q[{}], q[{}];\n'.format(qubit_pair_2[0], qubit_pair_2[1]))
        qasm.append(QasmUtils.partial_exchange_gate_qasm(-angle, qubit_pair_1[1], qubit_pair_2[0]))
        qasm.append(QasmUtils.partial_exchange_gate_qasm(-angle, qubit_pair_1[0], qubit_pair_2[1]))
        return ''.join(qasm)

    def get_qasm(self, var_parameters):
        assert len(var_parameters) == 1
        return self.double_exchange(var_parameters[0], self.qubit_pair_1, self.qubit_pair_2)


class ExchangeAnsatzBlock(AnsatzElement):
    def __init__(self, n_orbitals, n_electrons):
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons
        n_var_parameters = 2*int(n_orbitals/4) + 2*n_orbitals

        super(ExchangeAnsatzBlock, self).\
            __init__(element_type=str(self), n_var_parameters=n_var_parameters)

    def get_qasm(self, var_parameters):
        var_parameters_cycle = itertools.cycle(var_parameters)
        count = 0
        qasm = ['']
        # add single qubit n rotations
        for qubit in range(self.n_orbitals):
            qasm.append('rz ({}) q[{}];\n'.format(var_parameters_cycle.__next__(), qubit))
            count += 1
        # double orbital exchanges
        for qubit in range(self.n_orbitals):
            if qubit % 4 == 0:
                q_0 = qubit
                q_1 = (qubit + 1) % self.n_orbitals
                q_2 = (qubit + 2) % self.n_orbitals
                q_3 = (qubit + 3) % self.n_orbitals
                q_4 = (qubit + 4) % self.n_orbitals
                qasm.append(DoubleExchangeAnsatzElement.double_exchange(var_parameters_cycle.__next__(), [q_0, q_1], [q_2, q_3]))
                count += 1
                qasm.append(DoubleExchangeAnsatzElement.double_exchange(var_parameters_cycle.__next__(), [q_1, q_2], [q_3, q_4]))
                count += 1
        # single orbital exchanges
        for qubit in range(self.n_orbitals):
            if qubit % 2 == 0:
                qasm.append(QasmUtils.partial_exchange_gate_qasm(var_parameters_cycle.__next__(),
                                                                 qubit, (qubit + 1) % self.n_orbitals))
                count += 1
                qasm.append(QasmUtils.partial_exchange_gate_qasm(var_parameters_cycle.__next__(),
                                                                 (qubit+1) % self.n_orbitals, (qubit + 2) % self.n_orbitals))
                count += 1
        assert count == len(var_parameters)
        return ''.join(qasm)


# <<<<<<<<<<<<<<<<<<<<<<<<<< ansatzes (lists of ansatz elements) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# exchange single and double
class ESD:
    def __init__(self, n_orbitals, n_electrons):
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons

    def get_single_exchanges(self):
        single_excitations = []
        for i in range(self.n_electrons):
            for j in range(self.n_electrons, self.n_orbitals):
                single_excitations.append(ExchangeAnsatzElement(i, j))

        return single_excitations

    def get_double_exchanges(self):
        double_excitations = []
        for i in range(self.n_electrons - 1):
            for j in range(i + 1, self.n_electrons):
                for k in range(self.n_electrons, self.n_orbitals - 1):
                    for l in range(k + 1, self.n_orbitals):
                        double_excitations.append(DoubleExchangeAnsatzElement([i, j], [k, l]))

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
                fermi_operator = FermionOperator('[{1}^ {0}] - [{0}^ {1}]'.format(j, i))
                excitation = jordan_wigner(fermi_operator)
                single_excitations.append(AnsatzElement('excitation', excitation=excitation, element=fermi_operator,
                                                        excitation_order=1))
        return single_excitations

    def get_double_excitation_list(self):
        double_excitations = []
        for i in range(self.n_electrons-1):
            for j in range(i+1, self.n_electrons):
                for k in range(self.n_electrons, self.n_orbitals-1):
                    for l in range(k+1, self.n_orbitals):
                        fermi_operator = FermionOperator('[{2}^ {3}^ {0} {1}] - [{0}^ {1}^ {2} {3}]'.format(i, j, k, l))
                        excitation = jordan_wigner(fermi_operator)
                        double_excitations.append(AnsatzElement('excitation', excitation=excitation, element=fermi_operator,
                                                                excitation_order=2))
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



