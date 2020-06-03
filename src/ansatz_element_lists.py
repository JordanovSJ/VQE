from openfermion import QubitOperator, FermionOperator
from openfermion.transforms import jordan_wigner

from src.utils import QasmUtils, MatrixUtils
from src.ansatz_elements import*

import itertools
import numpy


# <<<<<<<<<<<<<<<<<<<<<<<<<< ansatzes (lists of ansatz elements) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class SDElements:
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
                    single_excitations.append(SingleFermiExcitation(i, j))
                elif self.element_type == 'qubit_excitation' or self.element_type == 'exchange':
                    single_excitations.append(SingleQubitExcitation(i, j))
                elif self.element_type == 'efficient_fermi_excitation':
                    single_excitations.append(EfficientSingleFermiExcitation(i, j))
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
                            double_excitations.append(DoubleFermiExcitation([i, j], [k, l]))
                        elif self.element_type == 'qubit_excitation':
                            double_excitations.append(DoubleQubitExcitation([i, j], [k, l]))
                        elif self.element_type == 'efficient_fermi_excitation':
                            double_excitations.append(EfficientDoubleFermiExcitation([i, j], [k, l]))
                        elif self.element_type == 'exchange':
                            double_excitations.append(DoubleExchange([i, j], [k, l], rescaled_parameter=self.rescaled,
                                                                     parity_dependence=self.parity_dependence,
                                                                     d_exc_correction=self.d_exc_correction))
                        else:
                            raise Exception('invalid double excitation type.')

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
                single_excitations.append(SingleQubitExcitation(i, j))

        return single_excitations

    def get_double_excitations(self):
        double_excitations = []
        for i in range(self.n_electrons - 1):
            for j in range(i + 1, self.n_electrons):
                for k in range(self.n_electrons, self.n_orbitals - 1):
                    for l in range(k + 1, self.n_orbitals):
                        if self.bosonic_excitation:
                            double_excitations.append(DoubleQubitExcitation([i, j], [k, l]))
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
            single_excitations.append(SingleQubitExcitation(*indices))

        return single_excitations

    def get_double_excitations(self):
        double_excitations = []
        for indices in itertools.combinations(range(self.n_orbitals), 4):
            if self.bosonic_excitation:
                double_excitations.append(DoubleQubitExcitation(indices[:2], indices[-2:]))
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
                single_excitations.append(SingleFermiExcitation(i, j))
        return single_excitations

    def get_double_excitations(self):
        double_excitations = []
        for i in range(self.n_electrons-1):
            for j in range(i+1, self.n_electrons):
                for k in range(self.n_electrons, self.n_orbitals-1):
                    for l in range(k+1, self.n_orbitals):
                        double_excitations.append(DoubleFermiExcitation([i, j], [k, l]))
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
                                                    order=1))
        return single_excitations

    def get_double_excitations(self):
        double_excitations = []
        for indices in itertools.combinations(range(self.n_orbitals), 4):
            fermi_operator = FermionOperator('[{2}^ {3}^ {0} {1}] - [{0}^ {1}^ {2} {3}]'.format(* indices))
            excitation = jordan_wigner(fermi_operator)
            double_excitations.append(AnsatzElement('excitation', excitation=excitation, element=fermi_operator,
                                                    order=2))
        return double_excitations

    def get_ansatz_elements(self):
        return self.get_single_excitations() + self.get_double_excitations()
