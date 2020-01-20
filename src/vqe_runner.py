from openfermion.transforms import get_fermion_operator, jordan_wigner, get_sparse_operator
from openfermionpsi4 import run_psi4
from openfermion.hamiltonians import MolecularData
from openfermion.utils import jw_hartree_fock_state
import src.backends as backends
from functools import partial

import scipy
import numpy
import time

import logging
from src.ansatz_types import UCCSD


class VQERunner:
    # Works for a single geometry
    def __init__(self, molecule, excitation_list=None, basis='sto-3g', molecule_geometry_params=None,
                 backend=backends.MatrixCalculation, initial_statevector=None):

        if molecule_geometry_params is None:
            molecule_geometry_params = {}

        self.iteration = None
        self.time = None

        self.molecule_name = molecule.name
        self.n_electrons = molecule.n_electrons
        self.n_orbitals = molecule.n_orbitals
        self.n_qubits = self.n_orbitals

        self.molecule_data = MolecularData(geometry=molecule.geometry(** molecule_geometry_params),
                                           basis=basis, multiplicity=molecule.multiplicity, charge=molecule.charge)
        # logging.info('Running VQE for geometry {}'.format(self.molecule_data.geometry))
        self.molecule_psi4 = run_psi4(self.molecule_data, run_mp2=False, run_cisd=False, run_ccsd=False, run_fci=False)

        # Get a qubit representation of the molecule hamiltonian
        self.molecule_ham = self.molecule_psi4.get_molecular_hamiltonian()
        self.fermion_ham = get_fermion_operator(self.molecule_ham)
        self.jw_ham_qubit_operator = jordan_wigner(self.fermion_ham)

        self.previous_energy = self.molecule_psi4.hf_energy.item()
        self.new_energy = None
        # logging.info('HF energy = {}'.format(self.energy))

        # get a list of excitations
        if excitation_list is None:
            self.excitation_list = UCCSD(self.n_orbitals, self.n_electrons).get_excitation_list()
        else:
            self.excitation_list = excitation_list

        self.backend = backend

        self.var_params = numpy.zeros(len(self.excitation_list))
        self.statevector = initial_statevector

    # Todo: a prettier way to write this?
    def get_energy(self, excitation_parameters, initial_statevector=None):
        energy, statevector, gate_counter = self.backend.get_energy(excitation_parameters=excitation_parameters,
                                                                    qubit_hamiltonian=self.jw_ham_qubit_operator,
                                                                    excitation_list=self.excitation_list,
                                                                    n_qubits=self.n_qubits,
                                                                    n_electrons=self.n_electrons,
                                                                    initial_statevector=initial_statevector)
        if statevector is not None:
            self.statevector = statevector

        self.new_energy = energy
        return energy

    # TODO: update to something useful
    def callback(self, xk):
        delta_e = self.new_energy - self.previous_energy
        self.previous_energy = self.new_energy

        print('Iteration: {}.\n Energy {}.  Energy change {}'.format(self.iteration, self.new_energy, '{:.3e}'.format(delta_e)))
        print('Iteration dutation: ', time.time() - self.time)
        self.time = time.time()
        self.iteration += 1
        # print('Excitaiton parameters :', xk)

    def vqe_run(self, max_n_iterations=None):

        if max_n_iterations is None:
            max_n_iterations = len(self.excitation_list) * 100

        print('-----Running VQE for: {}-----'.format(self.molecule_name))
        print('-----Number of electrons: {}-----'.format(self.n_electrons))
        print('-----Number of orbitals: {}-----'.format(self.n_orbitals))
        # print('-----Ansatz type {} ------'.format(self.ansatz))
        print('-----Numeber of excitation: {}-----'.format(len(self.excitation_list)))
        print('-----Statevector and energy calculate using {}------'.format(self.backend))

        excitation_parameters = numpy.zeros(len(self.excitation_list))

        self.iteration = 1
        self.time = time.time()
        opt_energy = scipy.optimize.minimize(self.get_energy, excitation_parameters, method='Nelder-Mead', callback=self.callback,
                                             options={'maxiter': max_n_iterations}, tol=1e-4)  # TODO: find a suitable optimizer

        return opt_energy
