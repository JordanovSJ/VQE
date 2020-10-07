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
        self.n_var_parameters = n_var_parameters
        self.element = element
        self.excitation_generator = excitation_generator
        self.excitation_matrix = None
        self.system_n_qubits = system_n_qubits

    def compute_excitation_mtrx(self):
        if self.excitation_generator is not None and self.system_n_qubits is not None:
            self.excitation_matrix = openfermion.get_sparse_operator(self.excitation_generator, n_qubits=self.system_n_qubits)

    # TODO Not sure if needed: It should be called to prevent memory leak when using multithreading with ray
    def delete_excitation_mtrx(self):
        self.excitation_matrix = None

    @staticmethod
    def get_qubit_excitation(qubits_1, qubits_2):

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


class PauliStringExc(AnsatzElement):
    def __init__(self, pauli_word_excitation, system_n_qubits=None):
        assert type(pauli_word_excitation) == QubitOperator
        assert len(pauli_word_excitation.terms) == 1
        assert list(pauli_word_excitation.terms.values())[0].real == 0  # it should be skew-Hermitian

        super(PauliStringExc, self).__init__(element=str(pauli_word_excitation),
                                             order=self.pauli_word_order(pauli_word_excitation),
                                             n_var_parameters=1, excitation_generator=pauli_word_excitation,
                                             system_n_qubits=system_n_qubits)

    @staticmethod
    def pauli_word_order(excitation):

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
        self.qubit_1 = qubit_1
        self.qubit_2 = qubit_2

        fermi_operator = FermionOperator('[{1}^ {0}] - [{0}^ {1}]'.format(self.qubit_2, self.qubit_1))
        excitation_generator = jordan_wigner(fermi_operator)

        super(SFExc, self).\
            __init__(element='s_f_exc_{}_{}'.format(qubit_1, qubit_2), order=1, n_var_parameters=1,
                     excitation_generator=excitation_generator, system_n_qubits=system_n_qubits)

    def get_qasm(self, var_parameters):
        assert len(var_parameters) == 1
        return QasmUtils.fermi_excitation(self.excitation_generator, var_parameters[0])


class DFExc(AnsatzElement):
    def __init__(self, qubit_pair_1, qubit_pair_2, system_n_qubits=None):
        self.qubit_pair_1 = qubit_pair_1
        self.qubit_pair_2 = qubit_pair_2

        fermi_operator = FermionOperator('[{2}^ {3}^ {0} {1}] - [{0}^ {1}^ {2} {3}]'
                                         .format(self.qubit_pair_1[0], self.qubit_pair_1[1],
                                                 self.qubit_pair_2[0], self.qubit_pair_2[1]))
        excitation_generator = jordan_wigner(fermi_operator)

        super(DFExc, self).\
            __init__(element='d_f_exc_{}_{}'.format(qubit_pair_1, qubit_pair_2), order=2, n_var_parameters=1,
                     excitation_generator=excitation_generator, system_n_qubits=system_n_qubits)

    def get_qasm(self, var_parameters):
        assert len(var_parameters) == 1

        return QasmUtils.fermi_excitation(self.excitation_generator, var_parameters[0])


class SQExc(AnsatzElement):
    def __init__(self, qubit_1, qubit_2, system_n_qubits=None):
        self.qubit_1 = qubit_1
        self.qubit_2 = qubit_2
        excitation_generator = self.get_qubit_excitation([qubit_1], [qubit_2])

        super(SQExc, self).\
            __init__(element='s_q_exc_{}_{}'.format(qubit_1, qubit_2), order=1, n_var_parameters=1,
                     excitation_generator=excitation_generator, system_n_qubits=system_n_qubits)

    def get_qasm(self, var_parameters):
        assert len(var_parameters) == 1
        return QasmUtils.partial_exchange(-var_parameters[0], self.qubit_1, self.qubit_2)  # the minus sign is important for consistence with the d_q_exc, as well obtaining the correct sign for grads...


class DQExc(AnsatzElement):
    def __init__(self, qubit_pair_1, qubit_pair_2, system_n_qubits=None):
        self.qubit_pair_1 = qubit_pair_1
        self.qubit_pair_2 = qubit_pair_2
        excitation_generator = self.get_qubit_excitation(qubit_pair_1, qubit_pair_2)

        super(DQExc, self).\
            __init__(element='d_q_exc_{}_{}'.format(qubit_pair_1, qubit_pair_2), order=2, n_var_parameters=1,
                     excitation_generator=excitation_generator, system_n_qubits=system_n_qubits)

    @staticmethod
    def d_q_exc_qasm(angle, qubit_pair_1_ref, qubit_pair_2_ref):
        # This is not required since the qubits are not ordered as for the fermi excitation
        qubit_pair_1 = qubit_pair_1_ref.copy()
        qubit_pair_2 = qubit_pair_2_ref.copy()

        angle = angle * 2  # for consistency with the conventional fermi excitation
        theta = angle / 8

        qasm = ['']

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

        return self.d_q_exc_qasm(parameter, self.qubit_pair_1, self.qubit_pair_2)


class EffSFExc(AnsatzElement):
    def __init__(self, qubit_1, qubit_2, system_n_qubits=None):
        self.qubit_1 = qubit_1
        self.qubit_2 = qubit_2

        fermi_operator = FermionOperator('[{1}^ {0}] - [{0}^ {1}]'.format(self.qubit_2, self.qubit_1))
        excitation_generator = jordan_wigner(fermi_operator)

        super(EffSFExc, self).\
            __init__(element='eff_s_f_exc_{}_{}'.format(qubit_2, qubit_1), order=1, n_var_parameters=1,
                     excitation_generator=excitation_generator, system_n_qubits=system_n_qubits)

    @staticmethod
    def eff_s_f_exc_qasm(angle, qubit_1, qubit_2):
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
        return self.eff_s_f_exc_qasm(var_parameters[0], self.qubit_1, self.qubit_2)


class EffDFExc(AnsatzElement):
    def __init__(self, qubit_pair_1, qubit_pair_2, system_n_qubits=None):
        self.qubit_pair_1 = qubit_pair_1
        self.qubit_pair_2 = qubit_pair_2

        fermi_operator = FermionOperator('[{2}^ {3}^ {0} {1}] - [{0}^ {1}^ {2} {3}]'
                                         .format(self.qubit_pair_1[0], self.qubit_pair_1[1],
                                                 self.qubit_pair_2[0], self.qubit_pair_2[1]))
        excitation_generator = jordan_wigner(fermi_operator)

        super(EffDFExc, self).\
            __init__(element='eff_d_f_exc_{}_{}'.format(qubit_pair_1, qubit_pair_2), order=2, n_var_parameters=1,
                     system_n_qubits=system_n_qubits, excitation_generator=excitation_generator)

    @staticmethod
    def eff_d_f_exc_qasm(angle, qubit_pair_1_ref, qubit_pair_2_ref):

        qubit_pair_1 = qubit_pair_1_ref.copy()
        qubit_pair_2 = qubit_pair_2_ref.copy()

        # !!!!!!!!! accounts for the missing functionality in the eff_d_f_exc circuit !!!!!
        if qubit_pair_1[0] > qubit_pair_1[1]:
            angle *= -1
        if qubit_pair_2[0] > qubit_pair_2[1]:
            angle *= -1

        angle = - angle * 2  # the factor of -2 is for consistency with the conventional fermi excitation
        theta = angle / 8

        qasm = ['']

        qubit_pair_1.sort()
        qubit_pair_2.sort()

        all_qubits = qubit_pair_1 + qubit_pair_2
        all_qubits.sort()

        # do not include the first qubits of qubit_pair_1 and qubit_pair_2
        # parity_qubits = list(range(qubit_pair_1[0] + 1, qubit_pair_1[1])) + list(range(qubit_pair_2[0] + 1, qubit_pair_2[1]))
        parity_qubits = list(range(all_qubits[0]+1, all_qubits[1])) + list(range(all_qubits[2]+1, all_qubits[3]))

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

        return self.eff_d_f_exc_qasm(parameter, self.qubit_pair_1, self.qubit_pair_2)


class SpinCompSFExc(AnsatzElement):
    def __init__(self, spin_orbital_1, spin_orbital_2, system_n_qubits=None):
        # TODO make this work with spin-orbitals
        # assert spatial_orbital_1 % 2 == 0
        # assert spatial_orbital_2 % 2 == 0

        # these are required for printing results only
        self.qubit_1 = spin_orbital_1
        self.qubit_2 = spin_orbital_2

        self.complement_qubit_1 = self.spin_complement_orbital(spin_orbital_1)
        self.complement_qubit_2 = self.spin_complement_orbital(spin_orbital_2)

        fermi_operator = FermionOperator('[{1}^ {0}] - [{0}^ {1}]'.format(spin_orbital_2, spin_orbital_1))
        fermi_operator += FermionOperator('[{1}^ {0}] - [{0}^ {1}]'.format(self.complement_qubit_2, self.complement_qubit_1))

        excitation_generator = jordan_wigner(fermi_operator)

        super(SpinCompSFExc, self).\
            __init__(element='spin_s_f_exc_{}_{}'.format(spin_orbital_2, spin_orbital_1), order=1, n_var_parameters=1,
                     excitation_generator=excitation_generator, system_n_qubits=system_n_qubits)

    def get_qasm(self, var_parameters):
        assert len(var_parameters) == 1
        return EffSFExc.eff_s_f_exc_qasm(var_parameters[0], self.qubit_1, self.qubit_2) \
               + EffSFExc.eff_s_f_exc_qasm(var_parameters[0], self.complement_qubit_1, self.complement_qubit_2)


class SpinCompDFExc(AnsatzElement):
    def __init__(self, orbitals_pair_1, orbitals_pair_2, system_n_qubits=None):

        # assert orbitals_pair_1[0] % 2 + orbitals_pair_1[1] % 2 == orbitals_pair_2[0] % 2 + orbitals_pair_2[1] % 2

        # these are required for printing results only
        self.qubit_pair_1 = orbitals_pair_1
        self.qubit_pair_2 = orbitals_pair_2

        self.orbitals_pair_1 = orbitals_pair_1
        self.orbitals_pair_2 = orbitals_pair_2

        self.complement_orbitals_pair_1 = [self.spin_complement_orbital(orbitals_pair_1[0]),
                                           self.spin_complement_orbital(orbitals_pair_1[1])]
        self.complement_orbitals_pair_2 = [self.spin_complement_orbital(orbitals_pair_2[0]),
                                           self.spin_complement_orbital(orbitals_pair_2[1])]

        fermi_operator = FermionOperator('[{2}^ {3}^ {0} {1}] - [{0}^ {1}^ {2} {3}]'
                                         .format(self.orbitals_pair_1[0], self.orbitals_pair_1[1],
                                                 self.orbitals_pair_2[0], self.orbitals_pair_2[1]))

        if set(self.orbitals_pair_1) != set(self.complement_orbitals_pair_1) or \
                set(self.orbitals_pair_2) != set(self.complement_orbitals_pair_2):

            fermi_operator += FermionOperator('[{2}^ {3}^ {0} {1}] - [{0}^ {1}^ {2} {3}]'
                                              .format(self.complement_orbitals_pair_1[0], self.complement_orbitals_pair_1[1],
                                                      self.complement_orbitals_pair_2[0], self.complement_orbitals_pair_2[1]))

        excitation_generator = jordan_wigner(fermi_operator)

        super(SpinCompDFExc, self).\
            __init__(element='spin_d_f_exc_{}_{}'.format(orbitals_pair_1, orbitals_pair_2), order=2, n_var_parameters=1,
                     system_n_qubits=system_n_qubits, excitation_generator=excitation_generator)

    def get_qasm(self, var_parameters):
        assert len(var_parameters) == 1
        parameter_1 = var_parameters[0]

        if set(self.orbitals_pair_1) == set(self.complement_orbitals_pair_1) and set(self.orbitals_pair_2) == set(self.complement_orbitals_pair_2):
            return EffDFExc.eff_d_f_exc_qasm(parameter_1, self.orbitals_pair_1, self.orbitals_pair_2)
        else:
            return EffDFExc.eff_d_f_exc_qasm(parameter_1, self.orbitals_pair_1, self.orbitals_pair_2) \
                   + EffDFExc.eff_d_f_exc_qasm(parameter_1, self.complement_orbitals_pair_1, self.complement_orbitals_pair_2)


class SpinCompSQExc(AnsatzElement):
    def __init__(self, spin_orbital_1, spin_orbital_2, sign=-1, system_n_qubits=None):

        # these are required for printing results only
        self.qubit_1 = spin_orbital_1
        self.qubit_2 = spin_orbital_2

        self.complement_qubit_1 = self.spin_complement_orbital(spin_orbital_1)
        self.complement_qubit_2 = self.spin_complement_orbital(spin_orbital_2)

        self.sign = sign

        if {self.qubit_1, self.qubit_2} == {self.complement_qubit_1, self.complement_qubit_2}:
            excitation_generator = self.get_qubit_excitation([spin_orbital_1], [spin_orbital_2])
        else:
            excitation_generator = self.get_qubit_excitation([spin_orbital_1], [spin_orbital_2])\
                         + self.sign*self.get_qubit_excitation([self.complement_qubit_1], [self.complement_qubit_2])

        super(SpinCompSQExc, self).\
            __init__(element='spin_s_q_exc_{}_{}'.format(spin_orbital_2, spin_orbital_1), order=1, n_var_parameters=1,
                     excitation_generator=excitation_generator, system_n_qubits=system_n_qubits)

    def get_qasm(self, var_parameters):
        assert len(var_parameters) == 1

        if {self.qubit_1, self.qubit_2} == {self.complement_qubit_1, self.complement_qubit_2}:
            return QasmUtils.partial_exchange(-var_parameters[0], self.qubit_1, self.qubit_2)
        else:
            # TODO adress the issues with the minus sign
            return QasmUtils.partial_exchange(-var_parameters[0], self.qubit_1, self.qubit_2) + \
                   QasmUtils.partial_exchange(-self.sign*var_parameters[0], self.complement_qubit_1, self.complement_qubit_2)


class SpinCompDQExc(AnsatzElement):
    def __init__(self, orbitals_pair_1, orbitals_pair_2, sign=-1, system_n_qubits=None):

        # TODO change names...
        # these are required for printing results only
        self.qubit_pair_1 = orbitals_pair_1
        self.qubit_pair_2 = orbitals_pair_2

        self.orbitals_pair_1 = orbitals_pair_1
        self.orbitals_pair_2 = orbitals_pair_2

        self.complement_orbitals_pair_1 = [self.spin_complement_orbital(orbitals_pair_1[0]),
                                           self.spin_complement_orbital(orbitals_pair_1[1])]
        self.complement_orbitals_pair_2 = [self.spin_complement_orbital(orbitals_pair_2[0]),
                                           self.spin_complement_orbital(orbitals_pair_2[1])]

        self.sign = sign

        if set(self.orbitals_pair_1) == set(self.complement_orbitals_pair_1) and \
           set(self.orbitals_pair_2) == set(self.complement_orbitals_pair_2):
            excitation_generator = self.get_qubit_excitation(self.orbitals_pair_1, self.orbitals_pair_2)
        else:
            # sign_1 = (-1)**((self.orbitals_pair_1[0] > self.orbitals_pair_1[1]) +
            #                 (self.orbitals_pair_2[0] > self.orbitals_pair_2[1]))
            # sign_2 = (-1) ** ((self.complement_orbitals_pair_1[0] > self.complement_orbitals_pair_1[1]) +
            #                   (self.complement_orbitals_pair_2[0] > self.complement_orbitals_pair_2[1]))

            excitation_generator = self.get_qubit_excitation(self.orbitals_pair_1, self.orbitals_pair_2) \
                         + self.sign*self.get_qubit_excitation(self.complement_orbitals_pair_1, self.complement_orbitals_pair_2)

        super(SpinCompDQExc, self).\
            __init__(element='spin_d_q_exc_{}_{}'.format(orbitals_pair_1, orbitals_pair_2), order=2, n_var_parameters=1,
                     system_n_qubits=system_n_qubits, excitation_generator=excitation_generator)

    def get_qasm(self, var_parameters):
        assert len(var_parameters) == 1
        parameter_1 = var_parameters[0]

        if set(self.orbitals_pair_1) == set(self.complement_orbitals_pair_1) and \
           set(self.orbitals_pair_2) == set(self.complement_orbitals_pair_2):
            return DQExc.d_q_exc_qasm(parameter_1, self.orbitals_pair_1, self.orbitals_pair_2)
        else:
            # sign_1 = (-1) ** ((self.orbitals_pair_1[0] > self.orbitals_pair_1[1]) +
            #                   (self.orbitals_pair_2[0] > self.orbitals_pair_2[1]))
            # sign_2 = (-1) ** ((self.complement_orbitals_pair_1[0] > self.complement_orbitals_pair_1[1]) +
            #                   (self.complement_orbitals_pair_2[0] > self.complement_orbitals_pair_2[1]))
            return DQExc.d_q_exc_qasm(parameter_1, self.orbitals_pair_1, self.orbitals_pair_2) \
                   + DQExc.d_q_exc_qasm(self.sign*parameter_1, self.complement_orbitals_pair_1, self.complement_orbitals_pair_2)

