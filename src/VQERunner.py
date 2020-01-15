from openfermion.transforms import get_fermion_operator, jordan_wigner, get_sparse_operator
from openfermionpsi4 import run_psi4
from openfermion.hamiltonians import MolecularData
from openfermion.utils import jw_hartree_fock_state
import src.backends as backends
from functools import partial

import scipy
import numpy

import logging
from src.AnsatzType import UCCSD


class VQERunner:
    # Works for a single geometry
    def __init__(self, molecule, excitation_list=None, basis='sto-3g', molecule_geometry_params=None, backend=backends.MatrixCalculation):

        if molecule_geometry_params is None:
            molecule_geometry_params = {}

        self.iter = None

        self.molecule_name = molecule.name
        self.n_electrons = molecule.n_electrons
        self.n_orbitals = molecule.n_orbitals
        self.n_qubits = self.n_orbitals

        self.molecule_data = MolecularData(geometry=molecule.geometry(** molecule_geometry_params),
                                           basis=basis, multiplicity=molecule.multiplicity, charge=molecule.charge)
        # logging.info('Running VQE for geometry {}'.format(self.molecule_data.geometry))

        self.molecule_psi4 = run_psi4(self.molecule_data, run_mp2=False, run_cisd=False, run_ccsd=False, run_fci=False)

        # Hamiltonian representations
        self.molecule_ham = self.molecule_psi4.get_molecular_hamiltonian()
        self.fermion_ham = get_fermion_operator(self.molecule_ham)
        self.jw_ham_qubit_operator = jordan_wigner(self.fermion_ham)

        self.previous_energy = self.molecule_psi4.hf_energy.item() # TODO: remove?
        self.new_energy = None
        # logging.info('HF energy = {}'.format(self.energy))

        if excitation_list is None:
            self.excitation_list = UCCSD(self.n_orbitals, self.n_electrons).get_excitation_list()
        else:
            self.excitation_list = excitation_list

        self.var_params = numpy.zeros(len(self.excitation_list))
        self.backend = backend

    # Todo: a prettier way to write this?
    def get_energy(self, excitation_parameters):
        energy = self.backend.get_energy(excitation_parameters=excitation_parameters,
                                         qubit_hamiltonian=self.jw_ham_qubit_operator,
                                         excitation_list=self.excitation_list,
                                         n_qubits=self.n_qubits,
                                         n_electrons=self.n_electrons)
        self.new_energy = energy
        return energy

    # TODO: update to something useful
    def callback(self, xk):
        delta_e = self.new_energy - self.previous_energy
        self.previous_energy = self.new_energy

        print('Iteration: {}. Energy change {}'.format(self.iter, '{:.3e}'.format(delta_e)))
        self.iter += 1
        # print('Excitaiton parameters :', xk)

    def vqe_run(self, max_n_iterations=None):

        if max_n_iterations is None:
            max_n_iterations = len(self.excitation_list) * 100

        print('-----Running VQE for {}-----'.format(self.molecule_name))
        print('-----Number of electrons {}-----'.format(self.n_electrons))
        print('-----Number of orbitals {}-----'.format(self.n_orbitals))

        # print('Number of excitations', len(self.excitation_list))

        # print('Qubit Hamiltonian: ', self.jw_ham_qubit_operator)

        parameters = numpy.zeros(len(self.excitation_list))

        self.iter = 1
        opt_energy = scipy.optimize.minimize(self.get_energy, parameters, method='Nelder-Mead', callback=self.callback,
                                             options={'maxiter': max_n_iterations}, tol=1e-5)  # TODO: find a suitable optimizer

        return opt_energy
