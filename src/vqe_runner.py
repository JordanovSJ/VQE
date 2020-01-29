from openfermion.transforms import get_fermion_operator, jordan_wigner, get_sparse_operator
from openfermionpsi4 import run_psi4
from openfermion.hamiltonians import MolecularData
from openfermion.utils import jw_hartree_fock_state

import src.backends as backends
from src.utils import QasmUtils

import scipy
import numpy
import time

import logging

from src.ansatz_elements import UCCSD


class VQERunner:
    # Works for a single geometry
    def __init__(self, molecule, ansatz_elements=None, basis='sto-3g', molecule_geometry_params=None,
                 backend=backends.QiskitSimulation, initial_statevector=None, optimizer=None, optimizer_options=None):

        if molecule_geometry_params is None:
            molecule_geometry_params = {}

        self.molecule_name = molecule.name
        self.n_electrons = molecule.n_electrons
        self.n_orbitals = molecule.n_orbitals
        self.n_qubits = self.n_orbitals

        self.molecule_data = MolecularData(geometry=molecule.geometry(** molecule_geometry_params),
                                           basis=basis, multiplicity=molecule.multiplicity, charge=molecule.charge)
        # logging.info('Running VQE for geometry {}'.format(self.molecule_data.geometry))
        self.molecule_psi4 = run_psi4(self.molecule_data, run_mp2=False, run_cisd=False, run_ccsd=False, run_fci=False)

        # Hamiltonian transforms
        self.molecule_ham = self.molecule_psi4.get_molecular_hamiltonian()
        self.fermion_ham = get_fermion_operator(self.molecule_ham)
        self.jw_ham_qubit_operator = jordan_wigner(self.fermion_ham)

        # ansatz_elements
        if ansatz_elements is None:
            self.ansatz_elements = UCCSD(self.n_orbitals, self.n_electrons).get_ansatz_elements()
        else:
            self.ansatz_elements = ansatz_elements

        self.var_parameters = numpy.zeros(sum([element.n_var_parameters for element in self.ansatz_elements]))
        self.statevector = initial_statevector

        # backend
        self.backend = backend
        self.optimizer = optimizer
        self.optimizer_options = optimizer_options

        # call back function variables
        self.previous_energy = self.molecule_psi4.hf_energy.item()
        self.new_energy = None
        self.iteration = None
        self.time = None
        self.gate_counter = None

    # Todo: a prettier way to write this?
    def get_energy(self, var_parameters, initial_statevector=None, update_gate_counter=False):
        energy, statevector, qasm = self.backend.get_energy(var_parameters=var_parameters,
                                                            qubit_hamiltonian=self.jw_ham_qubit_operator,
                                                            ansatz_elements=self.ansatz_elements,
                                                            n_qubits=self.n_qubits,
                                                            n_electrons=self.n_electrons,
                                                            initial_statevector=initial_statevector)
        if statevector is not None:
            self.statevector = statevector

        if update_gate_counter or self.iteration == 1:
            self.gate_counter = QasmUtils.gate_count(qasm, self.n_qubits)

        self.new_energy = energy
        # test
        print(energy)
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
            max_n_iterations = len(self.ansatz_elements) * 100

        print('-----Running VQE for: {}-----'.format(self.molecule_name))
        print('-----Number of electrons: {}-----'.format(self.n_electrons))
        print('-----Number of orbitals: {}-----'.format(self.n_orbitals))
        # print('-----Ansatz type {} ------'.format(self.ansatz))
        print('-----Numeber of excitation: {}-----'.format(len(self.ansatz_elements)))
        print('-----Statevector and energy calculate using {}------'.format(self.backend))

        var_parameters = self.var_parameters

        # <<<<<<<<<<<<<<, test >>>>>>>>>>>>>>>>..
        print('Test initial energy : ', self.get_energy(var_parameters))
        print(self.statevector)

        self.iteration = 1
        self.time = time.time()
        if self.optimizer is None:
            opt_energy = scipy.optimize.minimize(self.get_energy, var_parameters, method='L-BFGS-B', callback=self.callback,
                                                 options={'maxcor': 10, 'ftol': 1e-06, 'gtol': 1e-04,
                                                          'eps': 1e-04, 'maxfun': 1500, 'maxiter': max_n_iterations,
                                                          'iprint': -1, 'maxls': 5}, tol=1e-4)

            # opt_energy = scipy.optimize.minimize(self.get_energy, var_parameters, method='L-BFGS-B',
            #                                      callback=self.callback,
            #                                      options={'maxcor': 10, 'ftol': 1e-06, 'gtol': 1e-04,
            #                                               'eps': 1e-04, 'maxfun': 1500, 'maxiter': max_n_iterations,
            #                                               'iprint': -1, 'maxls': 5}, tol=1e-4)

            # opt_energy = scipy.optimize.minimize(self.get_energy, var_parameters, method='Powell', tol=1e-4,
            #                                      callback=self.callback)
        else:
            opt_energy = scipy.optimize.minimize(self.get_energy, var_parameters, method=self.optimizer,
                                                 options=self.optimizer_options, tol=1e-4, callback=self.callback)

        print('Gate counter', self.gate_counter)

        return opt_energy
