from openfermion.transforms import get_fermion_operator, jordan_wigner, get_sparse_operator
from openfermionpsi4 import run_psi4
from openfermion.hamiltonians import MolecularData

import src.backends as backends
from src.utils import QasmUtils, LogUtils
from src import config

import scipy
import numpy
import time
from functools import partial

import logging

from src.ansatz_element_lists import UCCSD
import ray


class VQERunner:
    # Works for a single geometry
    def __init__(self, molecule, ansatz_elements=None, basis='sto-3g', molecule_geometry_params=None,
                 backend=backends.QiskitSimulation, initial_statevector_qasm=None, optimizer=None,
                 optimizer_options=None, print_var_parameters=False, run_fci=True):

        LogUtils.vqe_info(molecule, ansatz_elements=ansatz_elements, basis=basis,
                          molecule_geometry_params=molecule_geometry_params, backend=backend)

        if molecule_geometry_params is None:
            molecule_geometry_params = {}

        self.molecule_name = molecule.name
        self.n_electrons = molecule.n_electrons
        self.n_orbitals = molecule.n_orbitals
        self.n_qubits = self.n_orbitals

        self.molecule_data = MolecularData(geometry=molecule.geometry(**molecule_geometry_params),
                                           basis=basis, multiplicity=molecule.multiplicity, charge=molecule.charge)
        self.molecule_psi4 = run_psi4(self.molecule_data, run_fci=run_fci)

        # Hamiltonian transforms
        self.molecule_ham = self.molecule_psi4.get_molecular_hamiltonian()
        self.fermion_ham = get_fermion_operator(self.molecule_ham)
        self.jw_ham_qubit_operator = jordan_wigner(self.fermion_ham)
        self.hf_energy = self.molecule_psi4.hf_energy.item()
        self.fci_energy = self.molecule_psi4.fci_energy.item()

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

        if optimizer is None:
            logging.info('Optimizer: {}. Optimizer options: {}'.format(config.optimizer, config.optimizer_options))
        else:
            logging.info('Optimizer: {}. Optimizer options: {}'.format(optimizer, optimizer_options))

        # call back function variables
        self.print_var_parameters = print_var_parameters
        self.previous_energy = self.hf_energy
        self.new_energy = None
        self.iteration = 0
        self.gate_counter = None

    def get_energy(self, var_parameters, ansatz_elements, multithread=False, initial_statevector_qasm=None,
                   update_gate_counter=False, multithread_iteration=None):
        t_start = time.time()
        # var_parameters = var_parameters[::-1]  # TODO cheat
        energy, statevector, qasm = self.backend.get_energy(var_parameters=var_parameters,
                                                            qubit_hamiltonian=self.jw_ham_qubit_operator,
                                                            ansatz_elements=ansatz_elements,
                                                            n_qubits=self.n_qubits,
                                                            n_electrons=self.n_electrons,
                                                            initial_statevector_qasm=initial_statevector_qasm)

        # if we run parallel process dont print and update info
        if multithread:
            if multithread_iteration is not None:
                try:
                    multithread_iteration[0] += 1
                except TypeError as te:
                    logging.warning(te)

            # TODO this logging does not work when running in parallel
            logging.info('Parallel process. Energy {}. Iteration duration: {}'.format(energy, time.time() - t_start))
        else:
            if update_gate_counter or self.iteration == 1:
                self.gate_counter = QasmUtils.gate_count_from_qasm(qasm, self.n_qubits)

            # print info
            self.new_energy = energy
            delta_e = self.new_energy - self.previous_energy
            self.previous_energy = self.new_energy

            message = 'Iteration: {}. Energy {}.  Energy change {} , Iteration dutation: {}' \
                .format(self.iteration, self.new_energy, '{:.3e}'.format(delta_e), time.time() - t_start)
            if self.print_var_parameters:
                message += ' Params: ' + str(var_parameters)
            logging.info(message)
            print(message)

            self.iteration += 1

        return energy

    def vqe_run(self, ansatz_elements=None, initial_var_parameters=None, initial_statevector_qasm=None):

        self.iteration = 1

        if ansatz_elements is None:
            var_parameters = self.var_parameters
            ansatz_elements = self.ansatz_elements
        else:
            if initial_var_parameters is None:
                var_parameters = numpy.zeros(sum([element.n_var_parameters for element in ansatz_elements]))
            else:
                assert len(initial_var_parameters) == sum([element.n_var_parameters for element in ansatz_elements])
                var_parameters = initial_var_parameters

        if initial_statevector_qasm is None:
            initial_statevector_qasm = self.initial_statevector_qasm

        # partial function to be used in the optimizer
        get_energy = partial(self.get_energy, ansatz_elements=ansatz_elements,
                             initial_statevector_qasm=initial_statevector_qasm)

        # if no ansatz elements supplied, calculate the energy without using the optimizer
        if len(ansatz_elements) == 0:
            return get_energy(var_parameters)

        message = ''
        message += '-----Running VQE for: {}-----\n'.format(self.molecule_name)
        message += '-----Number of electrons: {}-----\n'.format(self.n_electrons)
        message += '-----Number of orbitals: {}-----\n'.format(self.n_orbitals)
        message += '-----Numeber of ansatz elements: {}-----\n'.format(len(self.ansatz_elements))
        if len(ansatz_elements) == 1:
            message += '-----Ansatz type {}------\n'.format(ansatz_elements[0].element)
        message += '-----Statevector and energy calculated using {}------\n'.format(self.backend)
        message += '-----Optimizer {}------\n'.format(self.optimizer)
        print(message)
        logging.info(message)

        if self.optimizer is None:
            opt_energy = scipy.optimize.minimize(get_energy, var_parameters, method=config.optimizer,
                                                 options=config.optimizer_options, tol=config.optimizer_tol,
                                                 bounds=config.optimizer_bounds)
        else:
            opt_energy = scipy.optimize.minimize(get_energy, var_parameters, method=self.optimizer,
                                                 options=self.optimizer_options, tol=config.optimizer_tol,
                                                 bounds=config.optimizer_bounds)

        print(opt_energy)
        logging.info(opt_energy)
        print('Gate counter', self.gate_counter)
        logging.info('Gate counter' + str(self.gate_counter))

        opt_energy['n_iters'] = self.iteration  # cheating
        return opt_energy

    @ray.remote
    def vqe_run_multithread(self, ansatz_elements, initial_var_parameters=None, initial_statevector_qasm=None):

        if initial_var_parameters is None or initial_var_parameters == []:
            var_parameters = numpy.zeros(sum([element.n_var_parameters for element in ansatz_elements]))
        else:
            assert len(initial_var_parameters) == sum([element.n_var_parameters for element in ansatz_elements])
            var_parameters = initial_var_parameters

        # create it as a list so we can pass it by reference
        local_iteration = [0]

        # partial function to be used in the optimizer
        get_energy = partial(self.get_energy, ansatz_elements=ansatz_elements, multithread=True,
                             initial_statevector_qasm=initial_statevector_qasm, multithread_iteration=local_iteration)

        # if no ansatz elements supplied, calculate the energy without using the optimizer
        if len(ansatz_elements) == 0:
            return get_energy(var_parameters)

        if self.optimizer is None:
            opt_energy = scipy.optimize.minimize(get_energy, var_parameters, method=config.optimizer,
                                                 options=config.optimizer_options, tol=config.optimizer_tol,
                                                 bounds=config.optimizer_bounds)
        else:
            opt_energy = scipy.optimize.minimize(get_energy, var_parameters, method=self.optimizer,
                                                 options=self.optimizer_options, tol=config.optimizer_tol,
                                                 bounds=config.optimizer_bounds)

        if len(ansatz_elements) == 1:
            message = 'Ran VQE for element {}. Energy {}. Iterations {}'.format(ansatz_elements[0].element,
                                                                                opt_energy.fun, local_iteration[0])
            logging.info(message)
            print(message)
        else:
            message = 'Ran VQE. Energy {}. Iterations {}'.format(opt_energy.fun, local_iteration[0])
            logging.info(message)
            print(message)

        opt_energy['n_iters'] = local_iteration[0]  # cheating
        return opt_energy
