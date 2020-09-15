from openfermion import QubitOperator, FermionOperator
from openfermion.transforms import jordan_wigner

from src.utils import QasmUtils, MatrixUtils
from src.ansatz_elements import*

import itertools
import numpy


# <<<<<<<<<<<<<<<<<<<<<<<<<< ansatzes (lists of ansatz elements) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class SDExcitations:
    def __init__(self, n_orbitals, n_electrons, element_type='fermi_excitation', d_exchange_vars=None):
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons
        self.element_type = element_type

        if element_type == 'exchange':
            if d_exchange_vars is not None:
                self.rescaled = d_exchange_vars['rescaled']
                self.parity_dependence = d_exchange_vars['parity_dependence']
                self.d_exc_correction = d_exchange_vars['d_exc_correction']
            else:
                self.rescaled = False
                self.parity_dependence = False

    def get_single_excitations(self):
        single_excitations = []
        for i in range(self.n_electrons):
            for j in range(self.n_electrons, self.n_orbitals):
                if self.element_type == 'fermi_excitation':
                    single_excitations.append(SFExc(i, j, system_n_qubits=self.n_orbitals))
                elif self.element_type == 'qubit_excitation' or self.element_type == 'exchange':
                    single_excitations.append(SQExc(i, j, system_n_qubits=self.n_orbitals))
                elif self.element_type == 'efficient_fermi_excitation':
                    single_excitations.append(EffSFExc(i, j, system_n_qubits=self.n_orbitals))
                elif self.element_type == 'pauli_word_excitation':
                    qubit_excitation = SQExc(i, j).excitation
                    single_excitations += [PauliStringExc(1j * QubitOperator(term), system_n_qubits=self.n_orbitals) for term in
                                           qubit_excitation.terms]
                else:
                    raise Exception('Invalid single excitation type.')

        return single_excitations

    def get_double_excitations(self):
        double_excitations = []
        for i in range(self.n_electrons - 1):
            for j in range(i + 1, self.n_electrons):
                for k in range(self.n_electrons, self.n_orbitals - 1):
                    for l in range(k + 1, self.n_orbitals):
                        if self.element_type == 'fermi_excitation':
                            double_excitations.append(DFExc([i, j], [k, l],
                                                            system_n_qubits=self.n_orbitals))
                        elif self.element_type == 'qubit_excitation':
                            double_excitations.append(DQExc([i, j], [k, l],
                                                            system_n_qubits=self.n_orbitals))
                        elif self.element_type == 'efficient_fermi_excitation':
                            double_excitations.append(EffDFExc([i, j], [k, l]))
                        elif self.element_type == 'exchange':
                            double_excitations.append(DoubleExchange([i, j], [k, l], rescaled_parameter=self.rescaled,
                                                                     parity_dependence=self.parity_dependence,
                                                                     d_exc_correction=self.d_exc_correction))
                        elif self.element_type == 'pauli_word_excitation':
                            qubit_excitation = DQExc([i, j], [k, l]).excitation
                            double_excitations += [PauliStringExc(1j * QubitOperator(term), system_n_qubits=self.n_orbitals) for term in
                                                   qubit_excitation.terms]
                        else:
                            raise Exception('invalid double excitation type.')

        return double_excitations

    def get_ansatz_elements(self):
        return self.get_single_excitations() + self.get_double_excitations()


class GSDExcitations:
    def __init__(self, n_orbitals, n_electrons, element_type='fermi_excitation', d_exchange_vars=None):
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons
        self.element_type = element_type

        if element_type == 'exchange':
            if d_exchange_vars is not None:
                self.rescaled = d_exchange_vars['rescaled']
                self.parity_dependence = d_exchange_vars['parity_dependence']
                self.d_exc_correction = d_exchange_vars['d_exc_correction']
            else:
                self.rescaled = False
                self.parity_dependence = False

    def get_single_excitations(self):
        single_excitations = []
        for i, j in itertools.combinations(range(self.n_orbitals), 2):
            if self.element_type == 'fermi_excitation':
                single_excitations.append(SFExc(i, j, system_n_qubits=self.n_orbitals))
            elif self.element_type == 'qubit_excitation':
                single_excitations.append(SQExc(i, j, system_n_qubits=self.n_orbitals))
            elif self.element_type == 'efficient_fermi_excitation':
                single_excitations.append(EffSFExc(i, j, system_n_qubits=self.n_orbitals))
            elif self.element_type == 'pauli_word_excitation':
                qubit_excitation = SQExc(i, j).excitation
                single_excitations += [PauliStringExc(1j * QubitOperator(term), system_n_qubits=self.n_orbitals) for term in
                                       qubit_excitation.terms]
            else:
                raise Exception('Invalid single excitation type.')

        return single_excitations

    def get_double_excitations(self):
        double_excitations = []
        for i, j, k, l in itertools.combinations(range(self.n_orbitals), 4):
            if self.element_type == 'fermi_excitation':
                # double_excitations.append(DFExc([i, j], [k, l], system_n_qubits=self.n_orbitals))
                if i % 2 + j % 2 == k % 2 + l % 2:
                    double_excitations.append(DFExc([i, j], [k, l], system_n_qubits=self.n_orbitals))
                if i % 2 + k % 2 == j % 2 + l % 2:
                    double_excitations.append(DFExc([i, k], [j, l], system_n_qubits=self.n_orbitals))
                if i % 2 + l % 2 == k % 2 + j % 2:
                    double_excitations.append(DFExc([i, l], [k, j], system_n_qubits=self.n_orbitals))
            elif self.element_type == 'qubit_excitation':
                # double_excitations.append(DQExc([i, j], [k, l], system_n_qubits=self.n_orbitals))
                if i % 2 + j % 2 == k % 2 + l % 2:
                    double_excitations.append(DQExc([i, j], [k, l], system_n_qubits=self.n_orbitals))
                if i % 2 + k % 2 == j % 2 + l % 2:
                    double_excitations.append(DQExc([i, k], [j, l], system_n_qubits=self.n_orbitals))
                if i % 2 + l % 2 == k % 2 + j % 2:
                    double_excitations.append(DQExc([i, l], [k, j], system_n_qubits=self.n_orbitals))
            elif self.element_type == 'efficient_fermi_excitation':
                # double_excitations.append(EffDFExc([i, j], [k, l], system_n_qubits=self.n_orbitals))
                if i % 2 + j % 2 == k % 2 + l % 2:
                    double_excitations.append(EffDFExc([i, j], [k, l], system_n_qubits=self.n_orbitals))
                if i % 2 + k % 2 == j % 2 + l % 2:
                    double_excitations.append(EffDFExc([i, k], [j, l], system_n_qubits=self.n_orbitals))
                if i % 2 + l % 2 == k % 2 + j % 2:
                    double_excitations.append(EffDFExc([i, l], [k, j], system_n_qubits=self.n_orbitals))
            elif self.element_type == 'pauli_word_excitation':
                if (i + j) % 2 == (k + l) % 2:
                    qubit_excitation = DQExc([i, j], [k, l]).excitation
                    double_excitations += [PauliStringExc(1j * QubitOperator(term), system_n_qubits=self.n_orbitals) for term in
                                           qubit_excitation.terms]
            else:
                raise Exception('invalid double excitation type.')

        return double_excitations

    def get_ansatz_elements(self):
        return self.get_single_excitations() + self.get_double_excitations()


# Only for fermionic and qubit excitations
class SpinComplementGSDExcitations:
    def __init__(self, n_orbitals, n_electrons, element_type='fermi_excitation'):
        # works for spin zero systems only
        assert n_orbitals % 2 == 0
        assert n_electrons % 2 == 0

        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons
        self.element_type = element_type

    def get_single_excitations(self):
        single_excitations = []
        for i, j in itertools.combinations(range(int(self.n_orbitals)), 2):
            if self.element_type == 'fermi_excitation':
                new_s_excitation = SpinCompSFExc(i, j, system_n_qubits=self.n_orbitals)
            elif self.element_type == 'qubit_excitation':
                new_s_excitation = SpinCompSQExc(i, j, system_n_qubits=self.n_orbitals)
            else:
                raise Exception('invalid single spin complement excitation type.')

            if new_s_excitation.excitation != 0*openfermion.QubitOperator():
                single_excitations.append(new_s_excitation)

        return single_excitations

    def get_double_excitations(self):
        double_excitations = []

        for i, j, k, l in itertools.combinations(range(self.n_orbitals), 4):
            if i % 2 + j % 2 == k % 2 + l % 2:  # for PWE ?????? checked for H4

                if self.element_type == 'fermi_excitation':
                    new_d_excitation_1 = SpinCompDFExc([i, j], [k, l], system_n_qubits=self.n_orbitals)
                    new_d_excitation_2 = SpinCompDFExc([i, k], [j, l], system_n_qubits=self.n_orbitals)
                    new_d_excitation_3 = SpinCompDFExc([i, l], [k, j], system_n_qubits=self.n_orbitals)
                elif self.element_type == 'qubit_excitation':
                    new_d_excitation_1 = SpinCompDQExc([i, j], [k, l], system_n_qubits=self.n_orbitals)
                    new_d_excitation_2 = SpinCompDQExc([i, k], [j, l], system_n_qubits=self.n_orbitals)
                    new_d_excitation_3 = SpinCompDQExc([i, l], [k, j], system_n_qubits=self.n_orbitals)
                else:
                    raise Exception('invalid single spin complement excitation type.')

                if new_d_excitation_1.excitation != 0*openfermion.QubitOperator():
                    double_excitations.append(new_d_excitation_1)
                if new_d_excitation_2.excitation != 0*openfermion.QubitOperator():
                    double_excitations.append(new_d_excitation_2)
                if new_d_excitation_3.excitation != 0*openfermion.QubitOperator():
                    double_excitations.append(new_d_excitation_3)

        return double_excitations

    def get_ansatz_elements(self):
        return self.get_single_excitations() + self.get_double_excitations()


# exchange single and double
class ESD:
    def __init__(self, n_orbitals, n_electrons, rescaled=False, parity_dependence=False, d_exc_correction=False,
                 bosonic_excitation=False):
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons
        self.rescaled = rescaled
        self.parity_dependence = parity_dependence
        self.d_exc_correction = d_exc_correction
        self.bosonic_excitation = bosonic_excitation

    def get_single_excitations(self):
        single_excitations = []
        for i in range(self.n_electrons):
            for j in range(self.n_electrons, self.n_orbitals):
                single_excitations.append(SQExc(i, j))

        return single_excitations

    def get_double_excitations(self):
        double_excitations = []
        for i in range(self.n_electrons - 1):
            for j in range(i + 1, self.n_electrons):
                for k in range(self.n_electrons, self.n_orbitals - 1):
                    for l in range(k + 1, self.n_orbitals):
                        if self.bosonic_excitation:
                            double_excitations.append(DQExc([i, j], [k, l]))
                        else:
                            double_excitations.append(DoubleExchange([i, j], [k, l], rescaled_parameter=self.rescaled,
                                                                     parity_dependence=self.parity_dependence,
                                                                     d_exc_correction=self.d_exc_correction))

        return double_excitations

    def get_ansatz_elements(self):
        return self.get_single_excitations() + self.get_double_excitations()


class EGSD:
    def __init__(self, n_orbitals, n_electrons, rescaled=False, parity_dependence=False, d_exc_correction=False,
                 bosonic_excitation=False):
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons
        self.rescaled = rescaled
        self.parity_dependence = parity_dependence
        self.d_exc_correction = d_exc_correction
        self.bosonic_excitation = bosonic_excitation

    def get_single_excitations(self):
        single_excitations = []
        for indices in itertools.combinations(range(self.n_orbitals), 2):
            single_excitations.append(SQExc(*indices))

        return single_excitations

    def get_double_excitations(self):
        double_excitations = []
        for indices in itertools.combinations(range(self.n_orbitals), 4):
            if self.bosonic_excitation:
                double_excitations.append(DQExc(indices[:2], indices[-2:]))
            else:
                double_excitations.append(DoubleExchange(indices[:2], indices[-2:], rescaled_parameter=self.rescaled,
                                                         parity_dependence=self.parity_dependence,
                                                         d_exc_correction=self.d_exc_correction))

        return double_excitations

    def get_ansatz_elements(self):
        return self.get_single_excitations() + self.get_double_excitations()


class UCCSD:
    def __init__(self, n_orbitals, n_electrons):
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons

    def get_single_excitations(self):
        single_excitations = []
        for i in range(self.n_electrons):
            for j in range(self.n_electrons, self.n_orbitals):
                single_excitations.append(EffSFExc(i, j, system_n_qubits=self.n_orbitals))
        return single_excitations

    def get_double_excitations(self):
        double_excitations = []
        for i in range(self.n_electrons-1):
            for j in range(i+1, self.n_electrons):
                for k in range(self.n_electrons, self.n_orbitals-1):
                    for l in range(k+1, self.n_orbitals):
                        double_excitations.append(EffDFExc([i, j], [k, l], system_n_qubits=self.n_orbitals))
        return double_excitations

    def get_ansatz_elements(self):
        return self.get_single_excitations() + self.get_double_excitations()

#
# class UCCGSD:
#     def __init__(self, n_orbitals, n_electrons):
#         self.n_orbitals = n_orbitals
#         self.n_electrons = n_electrons
#
#     def get_single_excitations(self):
#         single_excitations = []
#         for indices in itertools.combinations(range(self.n_orbitals), 2):
#             fermi_operator = FermionOperator('[{1}^ {0}] - [{0}^ {1}]'.format(* indices))
#             excitation = jordan_wigner(fermi_operator)
#             single_excitations.append(AnsatzElement('excitation', excitation=excitation, element=fermi_operator,
#                                                     order=1))
#         return single_excitations
#
#     def get_double_excitations(self):
#         double_excitations = []
#         for indices in itertools.combinations(range(self.n_orbitals), 4):
#             fermi_operator = FermionOperator('[{2}^ {3}^ {0} {1}] - [{0}^ {1}^ {2} {3}]'.format(* indices))
#             excitation = jordan_wigner(fermi_operator)
#             double_excitations.append(AnsatzElement('excitation', excitation=excitation, element=fermi_operator,
#                                                     order=2))
#         return double_excitations
#
#     def get_ansatz_elements(self):
#         return self.get_single_excitations() + self.get_double_excitations()
