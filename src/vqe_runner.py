from openfermion.transforms import get_fermion_operator, jordan_wigner, get_sparse_operator
from openfermionpsi4 import run_psi4
from openfermion.hamiltonians import MolecularData

import src.backends as backends
from src.utils import QasmUtils, LogUtils

import scipy
import numpy
import time
from functools import partial

import logging

from src.ansatz_elements import UCCSD
import ray


class VQERunner:
    # Works for a single geometry
    def __init__(self, molecule, ansatz_elements=None, basis='sto-3g', molecule_geometry_params=None,
                 backend=backends.QiskitSimulation, initial_statevector_qasm=None, optimizer=None,
                 optimizer_options=None):
        LogUtils.vqe_info(molecule, ansatz_elements=ansatz_elements, basis=basis,
                          molecule_geometry_params=molecule_geometry_params, backend=backend)

        if molecule_geometry_params is None:
            molecule_geometry_params = {}

        self.molecule_name = molecule.name
        self.n_electrons = molecule.n_electrons
        self.n_orbitals = molecule.n_orbitals
        self.n_qubits = self.n_orbitals

        self.molecule_data = MolecularData(geometry=molecule.geometry(** molecule_geometry_params),
                                           basis=basis, multiplicity=molecule.multiplicity, charge=molecule.charge)
        self.molecule_psi4 = run_psi4(self.molecule_data)

        # Hamiltonian transforms
        self.molecule_ham = self.molecule_psi4.get_molecular_hamiltonian()
        self.fermion_ham = get_fermion_operator(self.molecule_ham)
        self.jw_ham_qubit_operator = jordan_wigner(self.fermion_ham)
        self.hf_energy = self.molecule_psi4.hf_energy.item()

        # ansatz_elements
        if ansatz_elements is None:
            self.ansatz_elements = []
        else:
            self.ansatz_elements = ansatz_elements

        self.var_parameters = numpy.zeros(sum([element.n_var_parameters for element in self.ansatz_elements]))
        # self.statevector = numpy.zeros(2**self.n_qubits)  # NOT used
        self.initial_statevector_qasm = initial_statevector_qasm

        # backend
        self.backend = backend
        self.optimizer = optimizer
        self.optimizer_options = optimizer_options

        # call back function variables
        self.previous_energy = self.hf_energy
        self.new_energy = None
        self.iteration = 0
        self.gate_counter = None

    def get_energy(self, var_parameters, ansatz_elements, multithread=False, initial_statevector_qasm=None,
                   update_gate_counter=False):
        t_start = time.time()
        energy, statevector, qasm = self.backend.get_energy(var_parameters=var_parameters,
                                                            qubit_hamiltonian=self.jw_ham_qubit_operator,
                                                            ansatz_elements=ansatz_elements,
                                                            n_qubits=self.n_qubits,
                                                            n_electrons=self.n_electrons,
                                                            initial_statevector_qasm=initial_statevector_qasm)

        # if we run parallel process dont print and update info
        if multithread:
            # TODO this logging does not work when running in parallel
            logging.info('Parallel process. Energy {}. Iteration duration: {}'.format(energy, time.time() - t_start))
        else:
            if update_gate_counter or self.iteration == 1:
                self.gate_counter = QasmUtils.gate_count(qasm, self.n_qubits)

            # print info
            self.new_energy = energy
            delta_e = self.new_energy - self.previous_energy
            self.previous_energy = self.new_energy

            message = 'Iteration: {}. Energy {}.  Energy change {} , Iteration dutation: {}'\
                .format(self.iteration, self.new_energy, '{:.3e}'.format(delta_e), time.time() - t_start)
            logging.info(message)
            print(message)
            self.iteration += 1

        return energy

    # Not used
    # def callback(self, xk):
    #     delta_e = self.new_energy - self.previous_energy
    #     self.previous_energy = self.new_energy
    #
    #     logging.info('Iteration: {}. Energy {}.  Energy change {} , Iteration dutation: {}'
    #                  .format(self.iteration, self.new_energy, '{:.3e}'.format(delta_e), time.time() - self.time))
    #     self.time = time.time()
    #     self.iteration += 1

    def vqe_run(self, ansatz_elements=None, initial_statevector_qasm=None, max_n_iterations=None):

        self.iteration = 1

        if max_n_iterations is None:
            max_n_iterations = len(self.ansatz_elements) * 100 + 100

        if ansatz_elements is None:
            var_parameters = self.var_parameters
            ansatz_elements = self.ansatz_elements
        else:
            var_parameters = numpy.zeros(sum([element.n_var_parameters for element in ansatz_elements]))

        if initial_statevector_qasm is None:
            initial_statevector_qasm = self.initial_statevector_qasm

        # partial function to be used in the optimizer
        get_energy = partial(self.get_energy, ansatz_elements=ansatz_elements,
                             initial_statevector_qasm=initial_statevector_qasm)

        # if no ansatz elements supplied, calculate the energy without using the optimizer
        if len(ansatz_elements) == 0:
            return get_energy(var_parameters)

        print('-----Running VQE for: {}-----'.format(self.molecule_name))
        print('-----Number of electrons: {}-----'.format(self.n_electrons))
        print('-----Number of orbitals: {}-----'.format(self.n_orbitals))
        print('-----Numeber of ansatz elements: {}-----'.format(len(self.ansatz_elements)))
        if len(ansatz_elements) == 1:
            print('-----Ansatz type {}------'.format(ansatz_elements[0].element_type))
        print('-----Statevector and energy calculated using {}------'.format(self.backend))
        print('-----Optimizer {}------'.format(self.optimizer))

        if self.optimizer is None:
            opt_energy = scipy.optimize.minimize(get_energy, var_parameters, method='L-BFGS-B',
                                                 options={'maxcor': 10, 'ftol': 1e-07, 'gtol': 1e-07,
                                                          'eps': 1e-04, 'maxfun': 1000, 'maxiter': max_n_iterations,
                                                          'iprint': -1, 'maxls': 10}, tol=1e-5)

            # # the comment code below is the most optimal set up for the optimizer so far
            # opt_energy = scipy.optimize.minimize(get_energy, var_parameters, method='L-BFGS-B',
            #                                                  options={'maxcor': 10, 'ftol': 1e-07, 'gtol': 1e-07,
            #                                                           'eps': 1e-04, 'maxfun': 1000, 'maxiter': max_n_iterations,
            #                                                           'iprint': -1, 'maxls': 10}, tol=1e-5)

        else:
            opt_energy = scipy.optimize.minimize(get_energy, var_parameters, method=self.optimizer,
                                                 options=self.optimizer_options, tol=1e-5)

        print(opt_energy)
        print('Gate counter', self.gate_counter)

        return opt_energy.fun

    @ray.remote
    def vqe_run_multithread(self, ansatz_elements, initial_statevector_qasm=None, max_n_iterations=None):

        if max_n_iterations is None:
            max_n_iterations = len(ansatz_elements) * 100

        var_parameters = numpy.zeros(sum([el.n_var_parameters for el in ansatz_elements]))

        # partial function to be used in the optimizer
        get_energy = partial(self.get_energy, ansatz_elements=ansatz_elements,
                             initial_statevector_qasm=initial_statevector_qasm, multithread=True)

        # if no ansatz elements supplied, calculate the energy without using the optimizer
        if len(ansatz_elements) == 0:
            return get_energy(var_parameters)
        
        if self.optimizer is None:
            opt_energy = scipy.optimize.minimize(get_energy, var_parameters, method='L-BFGS-B',
                                                 options={'maxcor': 10, 'ftol': 1e-07, 'gtol': 1e-07,
                                                          'eps': 1e-04, 'maxfun': 1000, 'maxiter': max_n_iterations,
                                                          'iprint': -1, 'maxls': 10}, tol=1e-5)

        else:
            opt_energy = scipy.optimize.minimize(get_energy, var_parameters, method=self.optimizer,
                                                 options=self.optimizer_options, tol=1e-5)

        if len(ansatz_elements) == 1:
            message = 'Ran VQE for ansatz_element {} . Energy {}'.format(ansatz_elements[0].fermi_operator, opt_energy.fun)
            logging.info(message)
            print(message)
        else:
            message = 'Ran VQE. Energy {}'.format(opt_energy.fun)
            logging.info(message)
            print(message)

        return opt_energy.fun
