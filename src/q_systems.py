from openfermion.hamiltonians import MolecularData
from openfermion import get_fermion_operator, freeze_orbitals, jordan_wigner, get_sparse_operator
from openfermionpsi4 import run_psi4

import numpy
import scipy
import pandas
import logging

from src.utils import MatrixUtils
from src.ansatz_elements import SQExc, DQExc
from src.iter_vqe_utils import DataUtils
from src.backends import QiskitSim


# basically a recipe to initialize some state
class State:
    def __init__(self, elements, parameters, n_qubits, n_electrons, init_state_qasm=None):
        self.n_qubits = n_qubits
        self.n_electrons = n_electrons
        self.elements = elements
        self.parameters = parameters
        self.init_state_qasm = init_state_qasm

    def add_element(self, element, parameter):
        self.elements.append(element)
        self.parameters.apppend(parameter)


class QSystem:

    def __init__(self, name, geometry, multiplicity, charge, n_orbitals, n_electrons, basis='sto-3g', frozen_els=None):
        self.name = name
        self.multiplicity = multiplicity
        self.charge = charge
        self.basis = basis
        self.geometry = geometry

        self.molecule_data = MolecularData(geometry=self.geometry, basis=basis, multiplicity=self.multiplicity,
                                           charge=self.charge)

        self.molecule_psi4 = run_psi4(self.molecule_data, run_fci=True)

        # Hamiltonian transforms
        self.molecule_ham = self.molecule_psi4.get_molecular_hamiltonian()
        self.hf_energy = self.molecule_psi4.hf_energy.item()
        self.fci_energy = self.molecule_psi4.fci_energy.item()
        self.energy_eigenvalues = None  # use this only if calculating excited states

        if frozen_els is None:
            self.n_electrons = n_electrons
            self.n_orbitals = n_orbitals
            self.n_qubits = n_orbitals
            self.fermion_ham = get_fermion_operator(self.molecule_ham)
        else:
            self.n_electrons = n_electrons - len(frozen_els['occupied'])
            self.n_orbitals = n_orbitals - len(frozen_els['occupied']) - len(frozen_els['unoccupied'])
            self.n_qubits = self.n_orbitals
            self.fermion_ham = freeze_orbitals(get_fermion_operator(self.molecule_ham), occupied=frozen_els['occupied'],
                                               unoccupied=frozen_els['unoccupied'], prune=True)
        self.jw_qubit_ham = jordan_wigner(self.fermion_ham)

        # this is used only for calculating excited states. list of [term_index, term_state]
        self.H_lower_state_terms = None

    #     self.H_sparse_matrix = None
    #     self.H_sparse_matrix_for_excited_state = None
    #
    # def calculate_ham_sparse_matrix(self, excited_state=1):
    #     self.H_sparse_matrix = get_sparse_operator(self.jw_qubit_ham)
    #     if self.H_lower_state_terms is not None:
    #         self.H_sparse_matrix_for_excited_state = QiskitSim.\
    #             ham_sparse_matrix_for_exc_state(self.H_sparse_matrix, self.H_lower_state_terms[:excited_state])

    # calculate the k smallest energy eigenvalues. For BeH2/H20 keep k<10 (too much memory)
    def calculate_energy_eigenvalues(self, k):
        H_sparse_matrix = get_sparse_operator(self.jw_qubit_ham)

        # do not calculate all eigenvectors of H, since this is very slow
        calculate_first_n = k + 1
        # use sigma to ensure we get the smallest eigenvalues
        eigvv = scipy.sparse.linalg.eigsh(H_sparse_matrix.todense(), k=calculate_first_n, which='SR')
        eigenvalues = list(eigvv[0])
        eigenvectors = list(eigvv[1].T)
        eigenvalues, eigenvectors = [*zip(*sorted([*zip(eigenvalues, eigenvectors)], key=lambda x:x[0]))]  # sort w.r.t. eigenvalues

        self.energy_eigenvalues = []

        i = 0
        while len(self.energy_eigenvalues) < k:

            if i >= len(eigenvalues):
                calculate_first_n += k
                eigvv = scipy.sparse.linalg.eigs(H_sparse_matrix.todense(), k=calculate_first_n, which='SR')
                eigenvalues = list(eigvv[0])
                eigenvectors = list(eigvv[1].T)
                eigenvalues, eigenvectors = [*zip(*sorted([*zip(eigenvalues, eigenvectors)], key=lambda x: x[0]))]

            if MatrixUtils.statevector_hamming_weight(eigenvectors[i].round(10)) == self.n_electrons:  # rounding set at random
                self.energy_eigenvalues.append(eigenvalues[i].real)

            i += 1
            if i == self.n_qubits**2:
                logging.warning('WARNING: Only {} eigenvalues found corresponding to the n_electrons'
                                .format(len(self.energy_eigenvalues)))
                break

        return self.energy_eigenvalues

    def set_h_lower_state_terms(self, states, factors=None):
        if factors is None:
            factors = list(numpy.zeros(len(states)) + abs(self.hf_energy)*2)  # default guess value

        self.H_lower_state_terms = [[factor, state] for factor, state in zip(factors, states)]


class H2(QSystem):

    def __init__(self, r=0.735, basis='sto-3g', frozen_els=None):
        super(H2, self).__init__(name='H2', geometry=self.get_geometry(r), multiplicity=1, charge=0, n_orbitals=4,
                                 n_electrons=2, basis=basis, frozen_els=frozen_els)

    # the ground and the first three degenerate excited states for H2 in equilibrium configuration
    # this is used for excited state simulations only
    def default_states(self):
        ground = State([DQExc([0, 1], [2, 3])], [0.11176849919227788], 4, 2)
        first_exc_1 = State([SQExc(0, 3)], [1.570796325683595], 4, 2)
        first_exc_2 = State([SQExc(1, 2)], [1.570796325683595], 4, 2)
        first_exc_3 = State([DQExc([0, 1], [2, 3]), SQExc(1, 3)], [-numpy.pi / 4, numpy.pi / 2], 4, 2)
        self.H_lower_state_terms = [[factor, state] for factor, state in
                                    zip([2.2, 1.55, 1.55, 1.55], [ground, first_exc_1, first_exc_2, first_exc_3])]
        return self.H_lower_state_terms

    @staticmethod
    def get_geometry(r=0.735):
        return [['H', [0, 0, 0]],
                ['H', [0, 0, r]]]


class H4(QSystem):

    def __init__(self, r=0.735, basis='sto-3g', frozen_els=None):
        super(H4, self).__init__(name='H4', geometry=self.get_geometry(r), multiplicity=1, charge=0, n_orbitals=8,
                                 n_electrons=4, basis=basis, frozen_els=frozen_els)

    @staticmethod
    def get_geometry(distance=0.735):
        return [
            ['H', [0, 0, 0]],
            ['H', [0, 0, distance]],
            ['H', [0, 0, 2 * distance]],
            ['H', [0, 0, 3 * distance]]
          ]


class LiH(QSystem):
    # frozen_els = {'occupied': [0,1], 'unoccupied': []}
    def __init__(self, r=1.546, basis='sto-3g', frozen_els=None):
        super(LiH, self).__init__(name='LiH', geometry=self.get_geometry(r), multiplicity=1, charge=0, n_orbitals=12,
                                  n_electrons=4, basis=basis, frozen_els=frozen_els)

    def default_states(self):
        df = pandas.read_csv('../../results/iter_vqe_results/vip/LiH_h_adapt_gsdqe_comp_pairs_15-Sep-2020.csv')
        ground = DataUtils.ansatz_from_data_frame(df, self)
        del df
        self.H_lower_state_terms = [[abs(self.hf_energy)*2, ground]]
        return self.H_lower_state_terms

    @staticmethod
    def get_geometry(r=1.546):
        return [['Li', [0, 0, 0]],
                ['H', [0, 0, r]]]


class HF(QSystem):

    def __init__(self, r=0.995, basis='sto-3g', frozen_els=None,):
        super(HF, self).__init__(name='HF', geometry=self.get_geometry(r), multiplicity=1, charge=0, n_orbitals=12,
                                 n_electrons=10, basis=basis, frozen_els=frozen_els,)

    @staticmethod
    def get_geometry(r=0.995):
        return [['F', [0, 0, 0]],
                ['H', [0, 0, r]]]


class BeH2(QSystem):
    # frozen_els = {'occupied': [0,1], 'unoccupied': [6,7]}
    def __init__(self, r=1.316, basis='sto-3g', frozen_els=None,):
        super(BeH2, self).__init__(name='BeH2', geometry=self.get_geometry(r), multiplicity=1, charge=0, n_orbitals=14,
                                   n_electrons=6, basis=basis, frozen_els=frozen_els,)

    @staticmethod
    def get_geometry(r=1.316):
        return [['Be', [0, 0, 0]],
                ['H', [0, 0, r]],
                ['H', [0, 0, -r]]]


class H2O(QSystem):

    def __init__(self, r=1.0285, theta=0.538*numpy.pi, basis='sto-3g', frozen_els=None):
        super(H2O, self).__init__(name='H20', geometry=self.get_geometry(r, theta), multiplicity=1, charge=0, n_orbitals=14,
                                  n_electrons=10, basis=basis, frozen_els=frozen_els,)

    @staticmethod
    def get_geometry(r=1.0285, theta=0.538 * numpy.pi):
        return [
            ['O', [0, 0, 0]],
            ['H', [0, 0, -r]],
            ['H', [0, r*numpy.sin(numpy.pi - theta), r*numpy.cos(numpy.pi - theta)]]
        ]



