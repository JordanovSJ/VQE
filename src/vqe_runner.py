from openfermion.transforms import get_fermion_operator, jordan_wigner, get_sparse_operator
from openfermionpsi4 import run_psi4
from openfermion.hamiltonians import MolecularData
from openfermion.utils import freeze_orbitals

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
    def __init__(self, q_system, ansatz_elements=None, backend=backends.QiskitSim, optimizer=config.optimizer,
                 optimizer_options=config.optimizer_options, print_var_parameters=False, use_ansatz_gradient=False):

        LogUtils.vqe_info(q_system, ansatz_elements=ansatz_elements, basis=q_system.basis,
                          molecule_geometry_params=q_system.get_geometry, backend=backend)

        self.q_system = q_system

        # ansatz_elements
        if ansatz_elements is None:
            self.ansatz_elements = []
        else:
            self.ansatz_elements = ansatz_elements

        self.var_parameters = numpy.zeros(sum([element.n_var_parameters for element in self.ansatz_elements]))

        # backend
        self.backend = backend
        self.optimizer = optimizer
        self.optimizer_options = optimizer_options
        self.use_ansatz_gradient = use_ansatz_gradient

        logging.info('Optimizer: {}. Optimizer options: {}'.format(optimizer, optimizer_options))

        # call back function variables
        self.print_var_parameters = print_var_parameters
        self.previous_energy = self.q_system.hf_energy
        self.new_energy = None
        self.iteration = 0
        self.gate_counter = None
        self.statevector = None

        # this is required for precomputing the statevector for ansatz_gradient. This works with the Quantum simulator only
        self.previous_var_parameters = None

        # ugly
        self.t_previous_iter = 0

    def get_energy(self, var_parameters, ansatz_elements, multithread=False, update_gate_counter=False,
                   multithread_iteration=None, initial_statevector_qasm=None, ham_matrix=None):

        if multithread is False:
            iteration_duration = time.time() - self.t_previous_iter
            self.t_previous_iter = time.time()

        energy, statevector, qasm = self.backend.get_expectation_value(q_system=self.q_system,
                                                                       var_parameters=var_parameters,
                                                                       ansatz_elements=ansatz_elements,
                                                                       initial_statevector_qasm=initial_statevector_qasm,
                                                                       operator_matrix=ham_matrix)

        self.statevector = statevector

        # TODO: the code below is a mess .. FIX
        # if we run in parallel process don't print and update info
        if multithread:
            if multithread_iteration is not None:
                try:
                    multithread_iteration[0] += 1
                except TypeError as te:
                    logging.warning(te)

            # # TODO this logging does not work when running in parallel
            # logging.info('Parallel process. Energy {}. Iteration duration: {}'.format(energy, time.time() - t_start))
        else:
            if update_gate_counter or self.iteration == 1:
                self.gate_counter = QasmUtils.gate_count_from_qasm(qasm, self.q_system.n_qubits)

            # print info
            self.new_energy = energy
            delta_e = self.new_energy - self.previous_energy
            self.previous_energy = self.new_energy

            message = 'Iteration: {}. Energy {}.  Energy change {} , Iteration dutation: {}' \
                .format(self.iteration, self.new_energy, '{:.3e}'.format(delta_e), iteration_duration)
            if self.print_var_parameters:
                message += ' Params: ' + str(var_parameters)
            logging.info(message)
            print(message)

            self.iteration += 1

        return energy

    def get_ansatz_gradient(self, var_parameters, ansatz_elements, ham_sparse_matrix, initial_statevector_qasm=None):
        return self.backend.ansatz_gradient(var_parameters, q_system=self.q_system, ansatz=ansatz_elements,
                                            init_state_qasm=initial_statevector_qasm,
                                            ham_sparse_matrix=ham_sparse_matrix)

    def vqe_run(self, ansatz_elements=None, initial_var_parameters=None, initial_statevector_qasm=None):

        self.iteration = 1

        # TODO: make the option to supply only initial_var_parameters valid
        if ansatz_elements is None:
            var_parameters = self.var_parameters
            ansatz_elements = self.ansatz_elements
        else:
            if initial_var_parameters is None:
                var_parameters = numpy.zeros(sum([element.n_var_parameters for element in ansatz_elements]))
            else:
                assert len(initial_var_parameters) == sum([element.n_var_parameters for element in ansatz_elements])
                var_parameters = initial_var_parameters

        # ugly
        self.t_previous_iter = time.time()

        ham_sparse_matrix = get_sparse_operator(self.q_system.jw_qubit_ham)
        ham_matrix = ham_sparse_matrix.todense()

        # partial function to be used in the optimizer
        get_energy = partial(self.get_energy, ansatz_elements=ansatz_elements,
                             initial_statevector_qasm=initial_statevector_qasm, ham_matrix=ham_matrix)

        get_gradient = partial(self.get_ansatz_gradient, ansatz_elements=ansatz_elements,
                               ham_sparse_matrix=ham_sparse_matrix,
                               initial_statevector_qasm=initial_statevector_qasm)

        # if no ansatz elements supplied, calculate the energy without using the optimizer
        if len(ansatz_elements) == 0:
            return get_energy(var_parameters)

        message = ''
        message += '-----Running VQE for: {}-----\n'.format(self.q_system.name)
        message += '-----Number of electrons: {}-----\n'.format(self.q_system.n_electrons)
        message += '-----Number of orbitals: {}-----\n'.format(self.q_system.n_orbitals)
        message += '-----Numeber of ansatz elements: {}-----\n'.format(len(ansatz_elements))
        if len(ansatz_elements) == 1:
            message += '-----Ansatz type {}------\n'.format(ansatz_elements[0].element)
        message += '-----Statevector and energy calculated using {}------\n'.format(self.backend)
        message += '-----Optimizer {}------\n'.format(self.optimizer)
        print(message)
        logging.info(message)

        if self.optimizer is None:
            self.optimizer = config.optimizer
            self.optimizer_options = config.optimizer_options

        if self.use_ansatz_gradient:
            opt_energy = scipy.optimize.minimize(get_energy, var_parameters, method=self.optimizer, jac=get_gradient,
                                                 options=self.optimizer_options, tol=config.optimizer_tol,
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

    # TODO add ansatz grad
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
        ham_sparse_matrix = get_sparse_operator(self.q_system.jw_qubit_ham)
        ham_matrix = ham_sparse_matrix.todense()

        get_energy = partial(self.get_energy, ansatz_elements=ansatz_elements, multithread=True,
                             multithread_iteration=local_iteration, ham_matrix=ham_matrix)

        get_gradient = partial(self.get_ansatz_gradient, ansatz_elements=ansatz_elements,
                               ham_sparse_matrix=ham_sparse_matrix,
                               initial_statevector_qasm=initial_statevector_qasm)

        # if no ansatz elements supplied, calculate the energy without using the optimizer
        if len(ansatz_elements) == 0:
            return get_energy(var_parameters)

        if self.optimizer is None:
            self.optimizer = config.optimizer
            self.optimizer_options = config.optimizer_options

        if self.use_ansatz_gradient:
            opt_energy = scipy.optimize.minimize(get_energy, var_parameters, method=self.optimizer, jac=get_gradient,
                                                 options=self.optimizer_options, tol=config.optimizer_tol,
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
