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
                    qubit_excitation = SQExc(i, j).excitation_generator
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
                            qubit_excitation = DQExc([i, j], [k, l]).excitation_generator
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
                qubit_excitation = SQExc(i, j).excitation_generator
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
                    qubit_excitation = DQExc([i, j], [k, l]).excitation_generator
                    double_excitations += [PauliStringExc(1j * QubitOperator(term), system_n_qubits=self.n_orbitals) for term in
                                           qubit_excitation.terms]
            else:
                raise Exception('invalid double excitation type.')

        return double_excitations

    def get_ansatz_elements(self):
        return self.get_single_excitations() + self.get_double_excitations()


# Only for fermionic and qubit excitations
class SpinComplementGSDExcitations:
    def __init__(self, n_orbitals, n_electrons, element_type='eff_fermi_excitation'):
        # works for spin zero systems only
        assert n_orbitals % 2 == 0
        assert n_electrons % 2 == 0

        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons
        self.element_type = element_type

    def get_single_excitations(self):
        single_excitations = []
        for i, j in itertools.combinations(range(int(self.n_orbitals)), 2):
            if self.element_type == 'efficient_fermi_excitation':
                new_s_excitation = SpinCompSFExc(i, j, system_n_qubits=self.n_orbitals)
                if new_s_excitation.excitation_generator != 0 * openfermion.QubitOperator():
                    single_excitations.append(new_s_excitation)
            elif self.element_type == 'qubit_excitation':
                new_s_excitation = SpinCompSQExc(i, j, sign=+1, system_n_qubits=self.n_orbitals)
                if new_s_excitation.excitation_generator != 0 * openfermion.QubitOperator():
                    single_excitations.append(new_s_excitation)
                new_s_excitation = SpinCompSQExc(i, j, sign=-1, system_n_qubits=self.n_orbitals)
                if new_s_excitation.excitation_generator != 0 * openfermion.QubitOperator():
                    single_excitations.append(new_s_excitation)
            else:
                raise Exception('invalid single spin complement excitation type.')

        return single_excitations

    def get_double_excitations(self):
        double_excitations = []

        for i, j, k, l in itertools.combinations(range(self.n_orbitals), 4):

            if self.element_type == 'efficient_fermi_excitation':
                new_d_excitation = SpinCompDFExc([i, j], [k, l], system_n_qubits=self.n_orbitals)
                if new_d_excitation.excitation_generator != 0 * openfermion.QubitOperator() and (
                        i % 2 + j % 2 == k % 2 + l % 2):
                    double_excitations.append(new_d_excitation)

                new_d_excitation = SpinCompDFExc([i, k], [j, l], system_n_qubits=self.n_orbitals)
                if new_d_excitation.excitation_generator != 0 * openfermion.QubitOperator() and (
                        i % 2 + k % 2 == j % 2 + l % 2):
                    double_excitations.append(new_d_excitation)

                new_d_excitation = SpinCompDFExc([i, l], [k, j], system_n_qubits=self.n_orbitals)
                if new_d_excitation.excitation_generator != 0 * openfermion.QubitOperator() and (
                        i % 2 + l % 2 == k % 2 + j % 2):
                    double_excitations.append(new_d_excitation)

            elif self.element_type == 'qubit_excitation':

                new_d_excitation = SpinCompDQExc([i, j], [k, l], sign=-1, system_n_qubits=self.n_orbitals)
                if new_d_excitation.excitation_generator != 0 * openfermion.QubitOperator() and (
                        i % 2 + j % 2 == k % 2 + l % 2):
                    double_excitations.append(new_d_excitation)
                new_d_excitation = SpinCompDQExc([i, j], [k, l], sign=+1, system_n_qubits=self.n_orbitals)
                if new_d_excitation.excitation_generator != 0 * openfermion.QubitOperator() and (
                        i % 2 + j % 2 == k % 2 + l % 2):
                    double_excitations.append(new_d_excitation)

                new_d_excitation = SpinCompDQExc([i, k], [j, l], sign=-1, system_n_qubits=self.n_orbitals)
                if new_d_excitation.excitation_generator != 0 * openfermion.QubitOperator() and (
                        i % 2 + k % 2 == j % 2 + l % 2):
                    double_excitations.append(new_d_excitation)
                new_d_excitation = SpinCompDQExc([i, k], [j, l], sign=+1, system_n_qubits=self.n_orbitals)
                if new_d_excitation.excitation_generator != 0 * openfermion.QubitOperator() and (
                        i % 2 + k % 2 == j % 2 + l % 2):
                    double_excitations.append(new_d_excitation)

                new_d_excitation = SpinCompDQExc([i, l], [k, j], sign=-1, system_n_qubits=self.n_orbitals)
                if new_d_excitation.excitation_generator != 0 * openfermion.QubitOperator() and (
                        i % 2 + l % 2 == k % 2 + j % 2):
                    double_excitations.append(new_d_excitation)
                new_d_excitation = SpinCompDQExc([i, l], [k, j], sign=+1, system_n_qubits=self.n_orbitals)
                if new_d_excitation.excitation_generator != 0 * openfermion.QubitOperator() and (
                        i % 2 + l % 2 == k % 2 + j % 2):
                    double_excitations.append(new_d_excitation)
            else:
                raise Exception('invalid single spin complement excitation type.')

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
                if i % 2 == j % 2:
                    single_excitations.append(EffSFExc(i, j, system_n_qubits=self.n_orbitals))
        return single_excitations

    def get_double_excitations(self):
        double_excitations = []
        for i in range(self.n_electrons-1):
            for j in range(i+1, self.n_electrons):
                for k in range(self.n_electrons, self.n_orbitals-1):
                    for l in range(k+1, self.n_orbitals):
                        if i % 2 + j % 2 == k % 2 + l % 2:
                            double_excitations.append(EffDFExc([i, j], [k, l], system_n_qubits=self.n_orbitals))
        return double_excitations

    def get_ansatz_elements(self):
        return self.get_single_excitations() + self.get_double_excitations()
