from openfermion import QubitOperator, FermionOperator
from openfermion.transforms import jordan_wigner
from openfermion.utils import hermitian_conjugated

from src.utils import QasmUtils, MatrixUtils

import openfermion
import itertools
import numpy


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< individual ansatz elements >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class AnsatzElement:
    def __init__(self, element, n_var_parameters=1, order=None, excitation_generator=None, system_n_qubits=None):
        self.order = order
        # self.qubits = qubits  # the qubits that define the ansatz element
        self.n_var_parameters = n_var_parameters
        self.element = element
        self.excitation_generator = excitation_generator
        self.excitation_matrix = None
        self.system_n_qubits = system_n_qubits

    def compute_excitation_mtrx(self):
        if self.excitation_generator is not None and self.system_n_qubits is not None:
            self.excitation_matrix = openfermion.get_sparse_operator(self.excitation_generator, n_qubits=self.system_n_qubits)

    def delete_excitation_mtrx(self):
        self.excitation_matrix = None

    @staticmethod
    def get_qubit_excitation_generator(qubits_1, qubits_2):

        assert len(qubits_2) == len(qubits_1)

        term_1 = QubitOperator('')

        for qubit_1, qubit_2 in zip(qubits_1, qubits_2):
            term_1 *= (QubitOperator('X{}'.format(qubit_1), 0.5) + QubitOperator('Y{}'.format(qubit_1), 0.5j))
            term_1 *= (QubitOperator('X{}'.format(qubit_2), 0.5) - QubitOperator('Y{}'.format(qubit_2), 0.5j))

        term_2 = hermitian_conjugated(term_1)

        return term_1 - term_2

    @staticmethod
    def spin_complement_orbital(orbital):
        if orbital % 2 == 0:
            return orbital + 1
        else:
            return orbital - 1

    @staticmethod
    def spin_complement_orbitals(orbitals):
        new_orbitals = []
        for orbital in orbitals:
            new_orbitals.append(AnsatzElement.spin_complement_orbital(orbital))
        return new_orbitals


class PauliStringExc(AnsatzElement):
    def __init__(self, pauli_word_excitation, system_n_qubits=None):
        assert type(pauli_word_excitation) == QubitOperator
        assert len(pauli_word_excitation.terms) == 1
        assert list(pauli_word_excitation.terms.values())[0].real == 0  # it should be skew-Hermitian

        super(PauliStringExc, self).__init__(element=str(pauli_word_excitation),
                                             order=self.pauli_string_order(pauli_word_excitation),
                                             n_var_parameters=1, excitation_generator=pauli_word_excitation,
                                             system_n_qubits=system_n_qubits)

    @staticmethod
    def pauli_string_order(excitation):

        pauli_ops = list(excitation.terms.keys())[0]
        order = 0

        for pauli_op in pauli_ops:
            assert pauli_op[1] in ['X', 'Y', 'Z']
            if pauli_op[1] != 'Z':
                order += 1

        return order

    def get_qasm(self, var_parameters):
        assert len(var_parameters) == 1
        return QasmUtils.fermi_excitation(self.excitation_generator, var_parameters[0])


class SFExc(AnsatzElement):
    def __init__(self, qubit_1, qubit_2, system_n_qubits=None):
        self.qubits = [[qubit_1], [qubit_2]]

        fermi_operator = FermionOperator('[{1}^ {0}] - [{0}^ {1}]'.format(self.qubits[1][0], self.qubits[0][0]))
        excitation_generator = jordan_wigner(fermi_operator)

        super(SFExc, self).\
            __init__(element='s_f_exc_{}_{}'.format(qubit_1, qubit_2), order=1, n_var_parameters=1,
                     excitation_generator=excitation_generator, system_n_qubits=system_n_qubits)

    def get_spin_comp_exc(self):
        return SFExc(self.spin_complement_orbital(self.qubits[0][0]), self.spin_complement_orbital(self.qubits[1][0]),
                     system_n_qubits=self.system_n_qubits)

    def get_qasm(self, var_parameters):
        assert len(var_parameters) == 1
        return QasmUtils.fermi_excitation(self.excitation_generator, var_parameters[0])


class DFExc(AnsatzElement):
    def __init__(self, qubit_pair_1, qubit_pair_2, system_n_qubits=None):
        assert len(qubit_pair_1) == 2
        assert len(qubit_pair_2) == 2
        self.qubits = [qubit_pair_1, qubit_pair_2]

        fermi_operator = FermionOperator('[{2}^ {3}^ {0} {1}] - [{0}^ {1}^ {2} {3}]'
                                         .format(self.qubits[0][0], self.qubits[0][1],
                                                 self.qubits[1][0], self.qubits[1][1]))
        excitation_generator = jordan_wigner(fermi_operator)

        super(DFExc, self).\
            __init__(element='d_f_exc_{}_{}'.format(qubit_pair_1, qubit_pair_2), order=2, n_var_parameters=1,
                     excitation_generator=excitation_generator, system_n_qubits=system_n_qubits)

    def get_spin_comp_exc(self):
        return DFExc(self.spin_complement_orbitals(self.qubits[0]), self.spin_complement_orbitals(self.qubits[1]),
                     system_n_qubits=self.system_n_qubits)

    def get_qasm(self, var_parameters):
        assert len(var_parameters) == 1

        return QasmUtils.fermi_excitation(self.excitation_generator, var_parameters[0])


class SQExc(AnsatzElement):
    def __init__(self, qubit_1, qubit_2, system_n_qubits=None):
        self.qubits = [[qubit_1], [qubit_2]]
        excitation_generator = self.get_qubit_excitation_generator([qubit_1], [qubit_2])

        super(SQExc, self).\
            __init__(element='s_q_exc_{}_{}'.format(qubit_1, qubit_2), order=1, n_var_parameters=1,
                     excitation_generator=excitation_generator, system_n_qubits=system_n_qubits)

    def get_spin_comp_exc(self):
        return SQExc(self.spin_complement_orbital(self.qubits[0][0]), self.spin_complement_orbital(self.qubits[1][0]),
                     system_n_qubits=self.system_n_qubits)

    def get_qasm(self, var_parameters):
        assert len(var_parameters) == 1
        return QasmUtils.partial_exchange(-var_parameters[0], self.qubits[0][0], self.qubits[1][0])  # the minus sign is important for consistence with the d_q_exc, as well obtaining the correct sign for grads...


class DQExc(AnsatzElement):
    def __init__(self, qubit_pair_1, qubit_pair_2, system_n_qubits=None):
        assert len(qubit_pair_1) == 2
        assert len(qubit_pair_2) == 2
        self.qubits = [qubit_pair_1, qubit_pair_2]
        excitation_generator = self.get_qubit_excitation_generator(qubit_pair_1, qubit_pair_2)

        super(DQExc, self).\
            __init__(element='d_q_exc_{}_{}'.format(qubit_pair_1, qubit_pair_2), order=2, n_var_parameters=1,
                     excitation_generator=excitation_generator, system_n_qubits=system_n_qubits)

    def get_spin_comp_exc(self):
        return DQExc(self.spin_complement_orbitals(self.qubits[0]), self.spin_complement_orbitals(self.qubits[1]),
                     system_n_qubits=self.system_n_qubits)

    def get_qasm(self, var_parameters):
        assert len(var_parameters) == 1
        parameter = var_parameters[0]

        return QasmUtils.d_q_exc_qasm(parameter, self.qubits[0], self.qubits[1])


class EffSFExc(AnsatzElement):
    def __init__(self, qubit_1, qubit_2, system_n_qubits=None):
        self.qubits = [[qubit_1], [qubit_2]]

        fermi_operator = FermionOperator('[{1}^ {0}] - [{0}^ {1}]'.format(self.qubits[1], self.qubits[0]))
        excitation_generator = jordan_wigner(fermi_operator)

        super(EffSFExc, self).\
            __init__(element='eff_s_f_exc_{}_{}'.format(qubit_2, qubit_1), order=1, n_var_parameters=1,
                     excitation_generator=excitation_generator, system_n_qubits=system_n_qubits)

    def get_spin_comp_exc(self):
        return EffSFExc(self.spin_complement_orbital(self.qubits[0][0]), self.spin_complement_orbital(self.qubits[1][0]),
                        system_n_qubits=self.system_n_qubits)

    def get_qasm(self, var_parameters):
        assert len(var_parameters) == 1
        return QasmUtils.eff_s_f_exc_qasm(var_parameters[0], self.qubits[0][0], self.qubits[1][0])


class EffDFExc(AnsatzElement):
    def __init__(self, qubit_pair_1, qubit_pair_2, system_n_qubits=None):
        assert len(qubit_pair_1) == 2
        assert len(qubit_pair_2) == 2
        self.qubits = [qubit_pair_1, qubit_pair_2]

        fermi_operator = FermionOperator('[{2}^ {3}^ {0} {1}] - [{0}^ {1}^ {2} {3}]'
                                         .format(self.qubits[0][0], self.qubits[0][1],
                                                 self.qubits[1][0], self.qubits[1][1]))
        excitation_generator = jordan_wigner(fermi_operator)

        super(EffDFExc, self).\
            __init__(element='eff_d_f_exc_{}_{}'.format(qubit_pair_1, qubit_pair_2), order=2, n_var_parameters=1,
                     system_n_qubits=system_n_qubits, excitation_generator=excitation_generator)

    def get_spin_comp_exc(self):
        return EffDFExc(self.spin_complement_orbitals(self.qubits[0]), self.spin_complement_orbitals(self.qubits[1]),
                        system_n_qubits=self.system_n_qubits)

    def get_qasm(self, var_parameters):
        assert len(var_parameters) == 1
        parameter = var_parameters[0]

        return QasmUtils.eff_d_f_exc_qasm(parameter, self.qubits[0], self.qubits[1])


class SpinCompSFExc(AnsatzElement):
    def __init__(self, qubit_1, qubit_2, system_n_qubits=None):
        self.qubits = [[qubit_1], [qubit_2]]
        self.complement_qubits = [self.spin_complement_orbitals([qubit_1]), self.spin_complement_orbitals([qubit_2])]

        fermi_operator = FermionOperator('[{1}^ {0}] - [{0}^ {1}]'.format(qubit_2, qubit_1))
        if {*self.qubits[0], *self.qubits[1]} != {*self.complement_qubits[0], *self.complement_qubits[1]} and \
           {*self.qubits[0], *self.qubits[1]} != {*self.complement_qubits[1], *self.complement_qubits[0]}:

            fermi_operator += FermionOperator('[{1}^ {0}] - [{0}^ {1}]'.format(self.complement_qubits[1][0],
                                                                               self.complement_qubits[0][0]))

        excitation_generator = jordan_wigner(fermi_operator)

        super(SpinCompSFExc, self).\
            __init__(element='spin_s_f_exc_{}_{}'.format(qubit_2, qubit_1), order=1, n_var_parameters=1,
                     excitation_generator=excitation_generator, system_n_qubits=system_n_qubits)

    def get_qasm(self, var_parameters):
        assert len(var_parameters) == 1

        qasm = QasmUtils.eff_s_f_exc_qasm(var_parameters[0], self.qubits[0][0], self.qubits[1][0])

        if {*self.qubits[0], *self.qubits[1]} != {*self.complement_qubits[0], *self.complement_qubits[1]} and \
           {*self.qubits[0], *self.qubits[1]} != {*self.complement_qubits[1], *self.complement_qubits[0]}:

            qasm += QasmUtils.eff_s_f_exc_qasm(var_parameters[0], self.complement_qubits[0][0], self.complement_qubits[1][0])

        return qasm


class SpinCompDFExc(AnsatzElement):
    def __init__(self, qubit_pair_1, qubit_pair_2, system_n_qubits=None):

        assert len(qubit_pair_1) == 2
        assert len(qubit_pair_2) == 2
        self.qubits = [qubit_pair_1, qubit_pair_2]
        self.complement_qubits = [self.spin_complement_orbitals(qubit_pair_1), self.spin_complement_orbitals(qubit_pair_2)]

        fermi_operator = FermionOperator('[{2}^ {3}^ {0} {1}] - [{0}^ {1}^ {2} {3}]'.format(*self.qubits[0], *self.qubits[1]))

        if [set(self.qubits[0]), set(self.qubits[1])] != [set(self.complement_qubits[0]), set(self.complement_qubits[1])] and \
           [set(self.qubits[0]), set(self.qubits[1])] != [set(self.complement_qubits[1]), set(self.complement_qubits[0])]:

            fermi_operator += FermionOperator('[{2}^ {3}^ {0} {1}] - [{0}^ {1}^ {2} {3}]'
                                              .format(*self.complement_qubits[0], *self.complement_qubits[1]))

        excitation_generator = jordan_wigner(fermi_operator)

        super(SpinCompDFExc, self).\
            __init__(element='spin_d_f_exc_{}_{}'.format(qubit_pair_1, qubit_pair_2), order=2, n_var_parameters=1,
                     system_n_qubits=system_n_qubits, excitation_generator=excitation_generator)

    def get_qasm(self, var_parameters):
        assert len(var_parameters) == 1
        parameter_1 = var_parameters[0]

        qasm = QasmUtils.eff_d_f_exc_qasm(parameter_1, self.qubits[0], self.qubits[1])

        # if the spin complement is different, add a qasm for it
        if [set(self.qubits[0]), set(self.qubits[1])] != [set(self.complement_qubits[0]), set(self.complement_qubits[1])] and \
           [set(self.qubits[0]), set(self.qubits[1])] != [set(self.complement_qubits[1]), set(self.complement_qubits[0])]:

           qasm += QasmUtils.eff_d_f_exc_qasm(parameter_1, self.complement_qubits[0], self.complement_qubits[1])

        return qasm


class SpinCompSQExc(AnsatzElement):
    def __init__(self, qubit_1, qubit_2, sign=-1, system_n_qubits=None):
        self.qubits = [[qubit_1], [qubit_2]]
        self.complement_qubits = [self.spin_complement_orbitals([qubit_1]), self.spin_complement_orbitals([qubit_2])]
        self.sign = sign

        if {*self.qubits[0], *self.qubits[1]} != {*self.complement_qubits[0], *self.complement_qubits[1]} and \
           {*self.qubits[0], *self.qubits[1]} != {*self.complement_qubits[1], *self.complement_qubits[0]}:
            excitation_generator = self.get_qubit_excitation_generator(self.qubits[0], self.qubits[1])
        else:
            excitation_generator = self.get_qubit_excitation_generator(self.qubits[0], self.qubits[1])\
                         + self.sign*self.get_qubit_excitation_generator(self.complement_qubits[0], self.complement_qubits[1])

        super(SpinCompSQExc, self).\
            __init__(element='spin_s_q_exc_{}_{}'.format(qubit_2, qubit_1), order=1, n_var_parameters=1,
                     excitation_generator=excitation_generator, system_n_qubits=system_n_qubits)

    def get_qasm(self, var_parameters):
        assert len(var_parameters) == 1

        qasm = QasmUtils.partial_exchange(-var_parameters[0], self.qubits[0][0], self.qubits[1][0])

        if {*self.qubits[0], *self.qubits[1]} != {*self.complement_qubits[0], *self.complement_qubits[1]} and \
           {*self.qubits[0], *self.qubits[1]} != {*self.complement_qubits[1], *self.complement_qubits[0]}:

            qasm += QasmUtils.partial_exchange(-self.sign*var_parameters[0], self.complement_qubits[0][0],
                                               self.complement_qubits[1][0])

        return qasm


class SpinCompDQExc(AnsatzElement):
    def __init__(self, qubit_pair_1, qubit_pair_2, sign=-1, system_n_qubits=None):

        assert len(qubit_pair_1) == 2
        assert len(qubit_pair_2) == 2
        self.qubits = [qubit_pair_1, qubit_pair_2]
        self.complement_qubits = [self.spin_complement_orbitals(qubit_pair_1),self.spin_complement_orbitals(qubit_pair_2)]

        self.sign = sign

        if [set(self.qubits[0]), set(self.qubits[1])] != [set(self.complement_qubits[0]), set(self.complement_qubits[1])] and \
           [set(self.qubits[0]), set(self.qubits[1])] != [set(self.complement_qubits[1]), set(self.complement_qubits[0])]:

            excitation_generator = self.get_qubit_excitation_generator(self.qubits[0], self.qubits[1])
        else:
            excitation_generator = self.get_qubit_excitation_generator(self.qubits[0], self.qubits[1]) \
                         + self.sign*self.get_qubit_excitation_generator(self.complement_qubits[0], self.complement_qubits[1])

        super(SpinCompDQExc, self).\
            __init__(element='spin_d_q_exc_{}_{}'.format(qubit_pair_1, qubit_pair_2), order=2, n_var_parameters=1,
                     system_n_qubits=system_n_qubits, excitation_generator=excitation_generator)

    def get_qasm(self, var_parameters):
        assert len(var_parameters) == 1
        parameter_1 = var_parameters[0]

        qasm = QasmUtils.d_q_exc_qasm(parameter_1, self.qubits[0], self.qubits[1])

        if [set(self.qubits[0]), set(self.qubits[1])] != [set(self.complement_qubits[0]), set(self.complement_qubits[1])] and \
           [set(self.qubits[0]), set(self.qubits[1])] != [set(self.complement_qubits[1]), set(self.complement_qubits[0])]:

            qasm += QasmUtils.d_q_exc_qasm(self.sign*parameter_1, self.complement_qubits[0], self.complement_qubits[1])

        return qasm

