# from openfermion.hamiltonians import MolecularData
from openfermion.chem import MolecularData
from openfermion import get_fermion_operator, freeze_orbitals, jordan_wigner, get_sparse_operator

#from openfermionpsi4 import run_psi4


import numpy
import scipy
import logging
import time

from src.utils import MatrixUtils


class QSystem:

    def __init__(self, name, geometry, multiplicity, charge, n_orbitals, n_electrons, basis='sto-3g', frozen_els=None):
        self.name = name
        self.multiplicity = multiplicity
        self.charge = charge
        self.basis = basis
        self.geometry = geometry

        self.molecule_data = MolecularData(geometry=self.geometry, basis=basis, multiplicity=self.multiplicity,
                                           charge=self.charge)

        # self.molecule_psi4 = run_psi4(self.molecule_data, run_fci=True)  # old version
        self.molecule_data.load()
        self.molecule_psi4 = self.molecule_data

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

    # calculate the k smallest energy eigenvalues. For BeH2/H20 keep k<10 (too much memory)
    def calculate_energy_eigenvalues(self, k):
        logging.info('Calculating excited states exact eigenvalues.')
        t0 = time.time()
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
        logging.info('Time: {}'.format(time.time() - t0))
        return self.energy_eigenvalues

    def set_h_lower_state_terms(self, states, factors=None):
        if factors is None:
            factors = list(numpy.zeros(len(states)) + abs(self.hf_energy)*2)  # default guess value

        self.H_lower_state_terms = [[factor, state] for factor, state in zip(factors, states)]

