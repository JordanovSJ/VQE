from openfermion import QubitOperator, FermionOperator
from openfermion.transforms import jordan_wigner

from src.utils import QasmUtils, MatrixUtils
from src.ansatz_elements import*

import itertools
import numpy


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Lists of ansatz elements>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class UCCSDExcitations:
    def __init__(self, n_orbitals, n_electrons, ansatz_element_type='f_exc'):
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons
        self.element_type = ansatz_element_type

    def get_single_excitations(self):
        single_excitations = []
        for i in range(self.n_electrons):
            for j in range(self.n_electrons, self.n_orbitals):
                if i % 2 == j % 2:
                    if self.element_type == 'f_exc':
                        single_excitations.append(SFExc(i, j, system_n_qubits=self.n_orbitals))
                    elif self.element_type == 'q_exc':
                        single_excitations.append(SQExc(i, j, system_n_qubits=self.n_orbitals))
                    elif self.element_type == 'eff_f_exc':
                        single_excitations.append(EffSFExc(i, j, system_n_qubits=self.n_orbitals))
                    elif self.element_type == 'pauli_str_exc':
                        qubit_excitation = SQExc(i, j).excitations_generators
                        single_excitations += [PauliStringExc(1j * QubitOperator(term), system_n_qubits=self.n_orbitals)
                                               for term in qubit_excitation.terms]
                    else:
                        raise Exception('Invalid single excitation type.')

        return single_excitations

    def get_double_excitations(self):
        double_excitations = []
        for i in range(self.n_electrons - 1):
            for j in range(i + 1, self.n_electrons):
                for k in range(self.n_electrons, self.n_orbitals - 1):
                    for l in range(k + 1, self.n_orbitals):
                        if i % 2 + j % 2 == k % 2 + l % 2:
                            if self.element_type == 'f_exc':
                                double_excitations.append(DFExc([i, j], [k, l],
                                                                system_n_qubits=self.n_orbitals))
                            elif self.element_type == 'q_exc':
                                double_excitations.append(DQExc([i, j], [k, l],
                                                                system_n_qubits=self.n_orbitals))
                            elif self.element_type == 'eff_f_exc':
                                double_excitations.append(EffDFExc([i, j], [k, l]))
                            elif self.element_type == 'pauli_str_exc':
                                qubit_excitation = DQExc([i, j], [k, l]).excitations_generators
                                double_excitations += [PauliStringExc(1j * QubitOperator(term), system_n_qubits=self.n_orbitals) for term in
                                                       qubit_excitation.terms]
                            else:
                                raise Exception('invalid double excitation type.')

        return double_excitations

    def get_excitations(self):
        return self.get_single_excitations() + self.get_double_excitations()


class SDExcitations:
    def __init__(self, n_orbitals, n_electrons, ansatz_element_type='f_exc', encoding='jw'):
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons
        self.ansatz_element_type = ansatz_element_type
        self.encoding = encoding

        if encoding != 'jw':
            assert encoding == 'bk'
            assert ansatz_element_type == 'f_exc'

    def get_single_excitations(self):
        single_excitations = []
        for i, j in itertools.combinations(range(self.n_orbitals), 2):
            # # test
            # if i % 2 == j % 2:
            if self.ansatz_element_type == 'f_exc':
                single_excitations.append(SFExc(i, j, system_n_qubits=self.n_orbitals, encoding=self.encoding))
            elif self.ansatz_element_type == 'q_exc':
                single_excitations.append(SQExc(i, j, system_n_qubits=self.n_orbitals))
            elif self.ansatz_element_type == 'eff_f_exc':
                single_excitations.append(EffSFExc(i, j, system_n_qubits=self.n_orbitals))
            elif self.ansatz_element_type == 'pauli_str_exc':
                excitations_generators = SQExc(i, j).excitations_generators
                for excitation_generator in excitations_generators:
                    single_excitations += [PauliStringExc(1j * QubitOperator(term), system_n_qubits=self.n_orbitals)
                                           for term in excitation_generator.terms]
            else:
                raise Exception('Invalid single excitation type.')

        return single_excitations

    def get_double_excitations(self):
        double_excitations = []
        for i, j, k, l in itertools.combinations(range(self.n_orbitals), 4):
            if self.ansatz_element_type == 'f_exc':
                if i % 2 + j % 2 == k % 2 + l % 2:
                    double_excitations.append(DFExc([i, j], [k, l], system_n_qubits=self.n_orbitals,
                                                    encoding=self.encoding))
                if i % 2 + k % 2 == j % 2 + l % 2:
                    double_excitations.append(DFExc([i, k], [j, l], system_n_qubits=self.n_orbitals,
                                                    encoding=self.encoding))
                if i % 2 + l % 2 == k % 2 + j % 2:
                    double_excitations.append(DFExc([i, l], [k, j], system_n_qubits=self.n_orbitals,
                                                    encoding=self.encoding))
            elif self.ansatz_element_type == 'q_exc':
                if i % 2 + j % 2 == k % 2 + l % 2:
                    double_excitations.append(DQExc([i, j], [k, l], system_n_qubits=self.n_orbitals))
                if i % 2 + k % 2 == j % 2 + l % 2:
                    double_excitations.append(DQExc([i, k], [j, l], system_n_qubits=self.n_orbitals))
                if i % 2 + l % 2 == k % 2 + j % 2:
                    double_excitations.append(DQExc([i, l], [k, j], system_n_qubits=self.n_orbitals))
            elif self.ansatz_element_type == 'eff_f_exc':
                if i % 2 + j % 2 == k % 2 + l % 2:
                    double_excitations.append(EffDFExc([i, j], [k, l], system_n_qubits=self.n_orbitals))
                if i % 2 + k % 2 == j % 2 + l % 2:
                    double_excitations.append(EffDFExc([i, k], [j, l], system_n_qubits=self.n_orbitals))
                if i % 2 + l % 2 == k % 2 + j % 2:
                    double_excitations.append(EffDFExc([i, l], [k, j], system_n_qubits=self.n_orbitals))
            elif self.ansatz_element_type == 'pauli_str_exc':
                if (i + j) % 2 == (k + l) % 2:
                    qubit_excitations = DQExc([i, j], [k, l]).excitations_generators
                    for excitation_generator in qubit_excitations:
                        double_excitations += [PauliStringExc(1j * QubitOperator(term), system_n_qubits=self.n_orbitals)
                                           for term in excitation_generator.terms]
            else:
                raise Exception('invalid double excitation type.')

        return double_excitations

    def get_excitations(self):
        return self.get_single_excitations() + self.get_double_excitations()


class GSDExcitations:
    def __init__(self, n_orbitals, n_electrons, ansatz_element_type='f_exc'):
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons
        self.ansatz_element_type = ansatz_element_type

    def get_single_excitations(self):
        single_excitations = []
        for i, j in itertools.combinations(range(self.n_orbitals), 2):
            # # test
            # if i % 2 == j % 2:
            if self.ansatz_element_type == 'f_exc':
                single_excitations.append(SFExc(i, j, system_n_qubits=self.n_orbitals))
            elif self.ansatz_element_type == 'q_exc':
                single_excitations.append(SQExc(i, j, system_n_qubits=self.n_orbitals))
            elif self.ansatz_element_type == 'eff_f_exc':
                single_excitations.append(EffSFExc(i, j, system_n_qubits=self.n_orbitals))
            elif self.ansatz_element_type == 'pauli_str_exc':
                excitations_generators = SQExc(i, j).excitations_generators
                for excitation_generator in excitations_generators:
                    single_excitations += [PauliStringExc(1j * QubitOperator(term), system_n_qubits=self.n_orbitals)
                                           for term in excitation_generator.terms]
            else:
                raise Exception('Invalid single excitation type.')

        return single_excitations

    def get_double_excitations(self):
        double_excitations = []
        for i, j, k, l in itertools.combinations(range(self.n_orbitals), 4):
            if self.ansatz_element_type == 'f_exc':
                double_excitations.append(DFExc([i, j], [k, l], system_n_qubits=self.n_orbitals))
                double_excitations.append(DFExc([i, k], [j, l], system_n_qubits=self.n_orbitals))
                double_excitations.append(DFExc([i, l], [k, j], system_n_qubits=self.n_orbitals))
            elif self.ansatz_element_type == 'q_exc':
                double_excitations.append(DQExc([i, j], [k, l], system_n_qubits=self.n_orbitals))
                double_excitations.append(DQExc([i, k], [j, l], system_n_qubits=self.n_orbitals))
                double_excitations.append(DQExc([i, l], [k, j], system_n_qubits=self.n_orbitals))
            elif self.ansatz_element_type == 'eff_f_exc':
                double_excitations.append(EffDFExc([i, j], [k, l], system_n_qubits=self.n_orbitals))
                double_excitations.append(EffDFExc([i, k], [j, l], system_n_qubits=self.n_orbitals))
                double_excitations.append(EffDFExc([i, l], [k, j], system_n_qubits=self.n_orbitals))
            elif self.ansatz_element_type == 'pauli_str_exc':
                qubit_excitations = DQExc([i, j], [k, l]).excitations_generators
                for excitation_generator in qubit_excitations:
                    double_excitations += [PauliStringExc(1j * QubitOperator(term), system_n_qubits=self.n_orbitals)
                                       for term in excitation_generator.terms]
            else:
                raise Exception('invalid double excitation type.')

        return double_excitations

    def get_excitations(self):
        return self.get_single_excitations() + self.get_double_excitations()


# Only for fermionic and qubit excitations, use for spin zero systems only
class SpinCompGSDExcitations:
    def __init__(self, n_orbitals, n_electrons, element_type='eff_f_exc', encoding='jw'):
        assert n_orbitals % 2 == 0
        assert n_electrons % 2 == 0
        if encoding != 'jw':
            assert encoding == 'bk'
            assert element_type == 'f_exc'

        self.encoding = encoding
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons
        self.element_type = element_type

    def get_single_excitations(self):
        single_excitations = []
        for i, j in itertools.combinations(range(int(self.n_orbitals)), 2):
            if self.element_type == 'eff_f_exc':
                new_s_excitation = SpinCompEffSFExc(i, j, system_n_qubits=self.n_orbitals)
                if new_s_excitation.excitations_generators != 0 * openfermion.QubitOperator():
                    single_excitations.append(new_s_excitation)
            elif self.element_type == 'f_exc':
                new_s_excitation = SpinCompSFExc(i, j, system_n_qubits=self.n_orbitals, encoding=self.encoding)
                if new_s_excitation.excitations_generators != 0 * openfermion.QubitOperator():
                    single_excitations.append(new_s_excitation)
            # qubit excitation does not work well
            elif self.element_type == 'q_exc':
                new_s_excitation = SpinCompSQExc(i, j, sign=+1, system_n_qubits=self.n_orbitals)
                if new_s_excitation.excitations_generators != 0 * openfermion.QubitOperator():
                    single_excitations.append(new_s_excitation)
                new_s_excitation = SpinCompSQExc(i, j, sign=-1, system_n_qubits=self.n_orbitals)
                if new_s_excitation.excitations_generators != 0 * openfermion.QubitOperator():
                    single_excitations.append(new_s_excitation)
            else:
                raise Exception('invalid single spin complement excitation type.')

        return single_excitations

    def get_double_excitations(self):
        double_excitations = []

        for i, j, k, l in itertools.combinations(range(self.n_orbitals), 4):

            if self.element_type == 'eff_f_exc':
                new_d_excitation = SpinCompEffDFExc([i, j], [k, l], system_n_qubits=self.n_orbitals)
                if new_d_excitation.excitations_generators != 0 * openfermion.QubitOperator() and (
                        i % 2 + j % 2 == k % 2 + l % 2):
                    double_excitations.append(new_d_excitation)

                new_d_excitation = SpinCompEffDFExc([i, k], [j, l], system_n_qubits=self.n_orbitals)
                if new_d_excitation.excitations_generators != 0 * openfermion.QubitOperator() and (
                        i % 2 + k % 2 == j % 2 + l % 2):
                    double_excitations.append(new_d_excitation)

                new_d_excitation = SpinCompEffDFExc([i, l], [k, j], system_n_qubits=self.n_orbitals)
                if new_d_excitation.excitations_generators != 0 * openfermion.QubitOperator() and (
                        i % 2 + l % 2 == k % 2 + j % 2):
                    double_excitations.append(new_d_excitation)

            elif self.element_type == 'f_exc':
                new_d_excitation = SpinCompDFExc([i, j], [k, l], system_n_qubits=self.n_orbitals, encoding=self.encoding)
                if new_d_excitation.excitations_generators != 0 * openfermion.QubitOperator() and (
                        i % 2 + j % 2 == k % 2 + l % 2):
                    double_excitations.append(new_d_excitation)

                new_d_excitation = SpinCompDFExc([i, k], [j, l], system_n_qubits=self.n_orbitals, encoding=self.encoding)
                if new_d_excitation.excitations_generators != 0 * openfermion.QubitOperator() and (
                        i % 2 + k % 2 == j % 2 + l % 2):
                    double_excitations.append(new_d_excitation)

                new_d_excitation = SpinCompDFExc([i, l], [k, j], system_n_qubits=self.n_orbitals, encoding=self.encoding)
                if new_d_excitation.excitations_generators != 0 * openfermion.QubitOperator() and (
                        i % 2 + l % 2 == k % 2 + j % 2):
                    double_excitations.append(new_d_excitation)

            elif self.element_type == 'q_exc':

                new_d_excitation = SpinCompDQExc([i, j], [k, l], sign=-1, system_n_qubits=self.n_orbitals)
                if new_d_excitation.excitations_generators != 0 * openfermion.QubitOperator() and (
                        i % 2 + j % 2 == k % 2 + l % 2):
                    double_excitations.append(new_d_excitation)
                new_d_excitation = SpinCompDQExc([i, j], [k, l], sign=+1, system_n_qubits=self.n_orbitals)
                if new_d_excitation.excitations_generators != 0 * openfermion.QubitOperator() and (
                        i % 2 + j % 2 == k % 2 + l % 2):
                    double_excitations.append(new_d_excitation)

                new_d_excitation = SpinCompDQExc([i, k], [j, l], sign=-1, system_n_qubits=self.n_orbitals)
                if new_d_excitation.excitations_generators != 0 * openfermion.QubitOperator() and (
                        i % 2 + k % 2 == j % 2 + l % 2):
                    double_excitations.append(new_d_excitation)
                new_d_excitation = SpinCompDQExc([i, k], [j, l], sign=+1, system_n_qubits=self.n_orbitals)
                if new_d_excitation.excitations_generators != 0 * openfermion.QubitOperator() and (
                        i % 2 + k % 2 == j % 2 + l % 2):
                    double_excitations.append(new_d_excitation)

                new_d_excitation = SpinCompDQExc([i, l], [k, j], sign=-1, system_n_qubits=self.n_orbitals)
                if new_d_excitation.excitations_generators != 0 * openfermion.QubitOperator() and (
                        i % 2 + l % 2 == k % 2 + j % 2):
                    double_excitations.append(new_d_excitation)
                new_d_excitation = SpinCompDQExc([i, l], [k, j], sign=+1, system_n_qubits=self.n_orbitals)
                if new_d_excitation.excitations_generators != 0 * openfermion.QubitOperator() and (
                        i % 2 + l % 2 == k % 2 + j % 2):
                    double_excitations.append(new_d_excitation)
            else:
                raise Exception('invalid single spin complement excitation type.')

        return double_excitations

    def get_excitations(self):
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

    def get_excitations(self):
        return self.get_single_excitations() + self.get_double_excitations()
