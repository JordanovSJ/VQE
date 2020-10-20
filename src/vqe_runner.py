from src.backends import QiskitSimCache, QiskitSim
from src.utils import LogUtils
from src import config

from openfermion import get_sparse_operator

import scipy
import numpy
import time
from functools import partial
import logging
import ray


# TODO make this class entirely static?
class VQERunner:
    # Works for a single geometry
    def __init__(self, q_system, backend=QiskitSim, optimizer=config.optimizer,
                 optimizer_options=config.optimizer_options, print_var_parameters=False, use_ansatz_gradient=False):

        self.backend = backend
        self.optimizer = optimizer
        self.optimizer_options = optimizer_options
        self.use_ansatz_gradient = use_ansatz_gradient
        self.print_var_parameters = print_var_parameters

        self.q_system = q_system

        self.previous_energy = self.q_system.hf_energy
        self.new_energy = None

        self.iteration = 0
        self.time_previous_iter = 0

    # TODO split this into a proper callback function!!!!!!
    def get_energy(self, var_parameters, ansatz, backend, multithread=False, multithread_iteration=None,
                   init_state_qasm=None, cache=None, excited_state=0):

        if multithread is False:
            iteration_duration = time.time() - self.time_previous_iter
            self.time_previous_iter = time.time()

        energy = backend.ham_expectation_value(self.q_system, ansatz, var_parameters, cache=cache,
                                               init_state_qasm=init_state_qasm, excited_state=excited_state)

        if multithread:
            if multithread_iteration is not None:
                try:
                    multithread_iteration[0] += 1
                except TypeError as te:
                    logging.warning(te)
        else:
            self.new_energy = energy
            delta_e = self.new_energy - self.previous_energy
            self.previous_energy = self.new_energy

            message = 'Iteration: {}. Energy {}.  Energy change {} , Iteration dutation: {}' \
                .format(self.iteration, self.new_energy, '{:.3e}'.format(delta_e), iteration_duration)
            if self.print_var_parameters:
                message += ' Params: ' + str(var_parameters)
            logging.info(message)

            self.iteration += 1

        return energy

    # def get_ansatz_gradient(self, var_parameters, ansatz, backend, init_state_qasm=None):
    #     return backend.ansatz_gradient(var_parameters, ansatz=ansatz, init_state_qasm=init_state_qasm)

    def vqe_run(self, ansatz, initial_var_parameters=None, init_state_qasm=None, excited_state=0, cache=None):

        assert len(ansatz) > 0
        if initial_var_parameters is None:
            var_parameters = numpy.zeros(sum([element.n_var_parameters for element in ansatz]))
        else:
            assert len(initial_var_parameters) == sum([element.n_var_parameters for element in ansatz])
            var_parameters = initial_var_parameters

        LogUtils.vqe_info(self.q_system, self.backend, self.optimizer, ansatz)

        self.iteration = 1
        self.time_previous_iter = time.time()

        # functions to be called by the optimizer
        get_energy = partial(self.get_energy, ansatz=ansatz, backend=self.backend, init_state_qasm=init_state_qasm,
                             excited_state=excited_state, cache=cache)

        get_gradient = partial(self.backend.ansatz_gradient, ansatz=ansatz, q_system=self.q_system,
                               init_state_qasm=init_state_qasm, cache=cache, excited_state=excited_state)

        if self.use_ansatz_gradient:
            result = scipy.optimize.minimize(get_energy, var_parameters, jac=get_gradient, method=self.optimizer,
                                             options=self.optimizer_options, tol=config.optimizer_tol,
                                             bounds=config.optimizer_bounds)
        else:

            result = scipy.optimize.minimize(get_energy, var_parameters, method=self.optimizer,
                                             options=self.optimizer_options, tol=config.optimizer_tol,
                                             bounds=config.optimizer_bounds)

        logging.info(result)

        result['n_iters'] = self.iteration  # cheating

        return result

    @ray.remote
    def vqe_run_multithread(self, ansatz, initial_var_parameters=None, init_state_qasm=None, excited_state=0, cache=None):

        assert len(ansatz) > 0

        if initial_var_parameters is None or initial_var_parameters == []:
            var_parameters = numpy.zeros(sum([element.n_var_parameters for element in ansatz]))
        else:
            assert len(initial_var_parameters) == sum([element.n_var_parameters for element in ansatz])
            var_parameters = initial_var_parameters

        # create it as a list so we can pass it by reference
        local_thread_iteration = [0]

        get_energy = partial(self.get_energy, ansatz=ansatz, backend=self.backend, init_state_qasm=init_state_qasm,
                             multithread=True, multithread_iteration=local_thread_iteration, cache=cache,
                             excited_state=excited_state)

        get_gradient = partial(self.backend.ansatz_gradient, ansatz=ansatz, init_state_qasm=init_state_qasm,
                               excited_state=excited_state, cache=cache, q_system=self.q_system)

        if self.use_ansatz_gradient:
            result = scipy.optimize.minimize(get_energy, var_parameters, method=self.optimizer, jac=get_gradient,
                                             options=self.optimizer_options, tol=config.optimizer_tol,
                                             bounds=config.optimizer_bounds)
        else:
            result = scipy.optimize.minimize(get_energy, var_parameters, method=self.optimizer,
                                             options=self.optimizer_options, tol=config.optimizer_tol,
                                             bounds=config.optimizer_bounds)

        # Logging does not work properly with ray multithreading. So use this printings. TODO: fix this. ..

        print('Ran VQE for last element {}. Energy {}. Iterations {}'.
              format(ansatz[-1].element, result.fun, local_thread_iteration[0]))

        # Not sure if needed
        del cache
        # if cache is not None:
        #     del cache.H_sparse_matrix_for_excited_state
        #     del cache.operator_sparse_matrix
        #     del cache.statevector
        #     del cache

        result['n_iters'] = local_thread_iteration[0]  # cheating

        return result

    # use this for QiskitSim only
    # @staticmethod
    @ray.remote
    def vqe_run_single_parameter_multithread(self, ansatz_element, cache):

        # create it as a list so we can pass it by reference
        local_thread_iteration = [0]

        operator_sparse_matrix = cache.H_sparse_matrix
        init_sparse_statevector = cache.init_sparse_statevector
        excitation_generator_matrix = cache.exc_gen_matrices[str(ansatz_element.excitation_generator)]
        commutator_sparse_matrix = cache.commutator_sparse_matrix

        # previous_parameter = None
        # sparse_statevector = init_sparse_statevector.copy()
        #
        # def update_statevector(parameter):
        #     if parameter != previous_parameter:
        #         previous_parameter = parameter
        #         sparse_statevector = scipy.sparse.linalg.expm_multiply(-parameter * excitation_generator_matrix,
        #                                                                init_sparse_statevector)

        # work only for excitations
        def get_energy(parameters):
            assert len(parameters) == 1
            parameter = parameters[0]
            sparse_statevector = scipy.sparse.linalg.expm_multiply(parameter * excitation_generator_matrix, init_sparse_statevector)
            energy = sparse_statevector.transpose().conj().dot(operator_sparse_matrix).dot(sparse_statevector).todense()[0, 0]
            del sparse_statevector
            local_thread_iteration[0] += 1
            return energy.real

        def get_gradient(parameters):
            assert len(parameters) == 1
            parameter = parameters[0]
            sparse_statevector = scipy.sparse.linalg.expm_multiply(parameter * excitation_generator_matrix, init_sparse_statevector)
            grad = sparse_statevector.transpose().conj().dot(commutator_sparse_matrix).dot(sparse_statevector).todense()[0, 0].real
            del sparse_statevector
            return numpy.array([grad])

        var_parameter = [0]
        if self.use_ansatz_gradient:
            result = scipy.optimize.minimize(get_energy, var_parameter, method=self.optimizer, jac=get_gradient,
                                             options=self.optimizer_options, tol=config.optimizer_tol,
                                             bounds=config.optimizer_bounds)
        else:
            result = scipy.optimize.minimize(get_energy, var_parameter, method=self.optimizer,
                                             options=self.optimizer_options, tol=config.optimizer_tol,
                                             bounds=config.optimizer_bounds)

        message = 'Ran VQE for element {}. Energy {}. Iterations {}'.format(ansatz_element.element, result.fun, local_thread_iteration[0])
        print(message)

        result['n_iters'] = local_thread_iteration[0]  # cheating

        # del operator_sparse_matrix
        # del init_sparse_statevector
        # del commutator_matrix

        return result

