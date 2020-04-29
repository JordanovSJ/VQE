from openfermion import QubitOperator, FermionOperator
from openfermion.transforms import jordan_wigner

from src.utils import QasmUtils, MatrixUtils
from src.ansatz_elements import*

import itertools
import numpy


# <<<<<<<<<<<<<<<<<<<<<<<<<< ansatzes (lists of ansatz elements) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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
                single_excitations.append(SingleExchange(i, j))

        return single_excitations

    def get_double_excitations(self):
        double_excitations = []
        for i in range(self.n_electrons - 1):
            for j in range(i + 1, self.n_electrons):
                for k in range(self.n_electrons, self.n_orbitals - 1):
                    for l in range(k + 1, self.n_orbitals):
                        if self.bosonic_excitation:
                            double_excitations.append(EfficientDoubleExcitation([i, j], [k, l]))
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
            single_excitations.append(SingleExchange(*indices))

        return single_excitations

    def get_double_excitations(self):
        double_excitations = []
        for indices in itertools.combinations(range(self.n_orbitals), 4):
            if self.bosonic_excitation:
                double_excitations.append(EfficientDoubleExcitation(indices[:2], indices[-2:]))
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
                single_excitations.append(SingleExcitation(i, j))
        return single_excitations

    def get_double_excitations(self):
        double_excitations = []
        for i in range(self.n_electrons-1):
            for j in range(i+1, self.n_electrons):
                for k in range(self.n_electrons, self.n_orbitals-1):
                    for l in range(k+1, self.n_orbitals):
                        double_excitations.append(DoubleExcitation([i, j], [k, l]))
        return double_excitations

    def get_ansatz_elements(self):
        return self.get_single_excitations() + self.get_double_excitations()


class UCCGSD:
    def __init__(self, n_orbitals, n_electrons):
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons

    def get_single_excitations(self):
        single_excitations = []
        for indices in itertools.combinations(range(self.n_orbitals), 2):
            fermi_operator = FermionOperator('[{1}^ {0}] - [{0}^ {1}]'.format(* indices))
            excitation = jordan_wigner(fermi_operator)
            single_excitations.append(AnsatzElement('excitation', excitation=excitation, element=fermi_operator,
                                                    excitation_order=1))
        return single_excitations

    def get_double_excitations(self):
        double_excitations = []
        for indices in itertools.combinations(range(self.n_orbitals), 4):
            fermi_operator = FermionOperator('[{2}^ {3}^ {0} {1}] - [{0}^ {1}^ {2} {3}]'.format(* indices))
            excitation = jordan_wigner(fermi_operator)
            double_excitations.append(AnsatzElement('excitation', excitation=excitation, element=fermi_operator,
                                                    excitation_order=2))
        return double_excitations

    def get_ansatz_elements(self):
        return self.get_single_excitations() + self.get_double_excitations()
