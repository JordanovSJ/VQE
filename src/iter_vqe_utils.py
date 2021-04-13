from openfermion import get_sparse_operator, QubitOperator
from src import config
from src import backends
from src.ansatz_elements import *
from src.utils import QasmUtils
from src.state import State

from src.backends import QiskitSimBackend
from scripts.zhenghao.noisy_backends import QasmBackend

import time
import ray
import ast
import numpy
import scipy


class IterVQEQasmUtils:
    @staticmethod
    def gate_count_from_ansatz(ansatz, n_qubits, var_parameters=None):
        n_var_parameters = sum([x.n_var_parameters for x in ansatz])
        if var_parameters is None:
            var_parameters = numpy.zeros(n_var_parameters)
        else:
            assert n_var_parameters == len(var_parameters)
        qasm = backends.QiskitSimBackend.qasm_from_ansatz(ansatz, var_parameters)
        return QasmUtils.gate_count_from_qasm(qasm, n_qubits)


class EnergyUtils:
    # calculate the full (optimizing all parameters) VQE energy reductions for a set of ansatz elements
    @staticmethod
    def elements_full_vqe_energy_reductions(vqe_runner, ansatz_elements, elements_parameters=None, ansatz=None,
                                            ansatz_parameters=None, global_cache=None, excited_state=0):

        if ansatz is None:
            ansatz = []
            ansatz_parameters = []

        # TODO this will work only if the ansatz element has 1 var. par.
        if elements_parameters is None:
            elements_parameters = list(numpy.zeros(len(ansatz_elements)))
            print(elements_parameters)

        def get_thread_cache():
            if global_cache is not None:
                return global_cache.get_vqe_thread_cache()
            else:
                return None

        if config.multithread:
            ray.init(num_cpus=config.ray_options['n_cpus'])
            elements_ray_ids = [
                [element,
                 vqe_runner.vqe_run_multithread.remote(self=vqe_runner, ansatz=ansatz + [element],
                                                       init_guess_parameters=ansatz_parameters + [
                                                           elements_parameters[i]],
                                                       cache=get_thread_cache(), excited_state=excited_state)]
                for i, element in enumerate(ansatz_elements)
            ]
            elements_results = [[element_ray_id[0], ray.get(element_ray_id[1])] for element_ray_id in elements_ray_ids]
            ray.shutdown()
        else:
            elements_results = [
                [element, vqe_runner.vqe_run(ansatz=ansatz + [element], excited_state=excited_state,
                                             init_guess_parameters=ansatz_parameters + [elements_parameters[i]],
                                             cache=global_cache)]
                for i, element in enumerate(ansatz_elements)
            ]

        return elements_results

    # returns the ansatz element that achieves the largest full (optimizing all parameters) VQE energy reduction
    @staticmethod
    def largest_full_vqe_energy_reduction_element(vqe_runner, ansatz_elements, elements_parameters=None, ansatz=None,
                                                  ansatz_parameters=None, global_cache=None, excited_state=0):
        elements_results = EnergyUtils.elements_full_vqe_energy_reductions(vqe_runner, ansatz_elements,
                                                                           elements_parameters=elements_parameters,
                                                                           ansatz=ansatz,
                                                                           ansatz_parameters=ansatz_parameters,
                                                                           excited_state=excited_state,
                                                                           global_cache=global_cache)
        # return min(elements_results, key=lambda x: x[1].fun)
        elements_results.sort(key=lambda x: x[1].fun)
        return elements_results[0]

    # calculate the full (optimizing all parameters) VQE energy reductions for a set of ansatz elements
    @staticmethod
    def elements_individual_vqe_energy_reductions(vqe_runner, ansatz_elements, elements_parameters=None, ansatz=None,
                                                  ansatz_parameters=None, excited_state=0, global_cache=None):

        if ansatz is None:
            ansatz = []
            ansatz_parameters = []

        # TODO this will work only if the ansatz element has 1 var. par.
        if elements_parameters is None:
            elements_parameters = list(numpy.zeros(len(ansatz_elements)))

        if vqe_runner.backend == backends.QiskitSimBackend:
            ansatz_qasm = QasmUtils.hf_state(vqe_runner.q_system.n_electrons)
            ansatz_qasm += vqe_runner.backend.qasm_from_ansatz(ansatz, ansatz_parameters)
        else:
            ansatz_qasm = None

        def get_thread_cache(element):
            if global_cache is not None:
                init_sparse_statevector = global_cache.get_statevector(ansatz, ansatz_parameters)
                return global_cache.single_par_vqe_thread_cache(element, init_sparse_statevector)
            else:
                return None

        if config.multithread:
            ray.init(num_cpus=config.ray_options['n_cpus'])
            elements_ray_ids = [
                [element,
                 vqe_runner.vqe_run_multithread.remote(self=vqe_runner, ansatz=[element], init_state_qasm=ansatz_qasm,
                                                       init_guess_parameters=[elements_parameters[i]],
                                                       excited_state=excited_state, cache=get_thread_cache(element))
                 ]
                # TODO this will work only if the ansatz element has 1 var. par.
                for i, element in enumerate(ansatz_elements)
            ]
            elements_results = [[element_ray_id[0], ray.get(element_ray_id[1])] for element_ray_id in
                                elements_ray_ids]
            ray.shutdown()
        else:
            # use thread cache even if not multithreading since it contains the precalculated init_sparse_statevector
            elements_results = [
                [element, vqe_runner.vqe_run(ansatz=[element], init_guess_parameters=[elements_parameters[i]],
                                             excited_state=excited_state, init_state_qasm=ansatz_qasm,
                                             cache=get_thread_cache(element))
                 ]
                for i, element in enumerate(ansatz_elements)
            ]

        return elements_results

    # returns the ansatz element that achieves the largest full (optimizing all parameters) VQE energy reduction
    @staticmethod
    def largest_individual_vqe_energy_reduction_elements(vqe_runner, elements, elements_parameters=None, ansatz=None,
                                                         ansatz_parameters=None, global_cache=None, n=1,
                                                         excited_state=0):

        elements_results = EnergyUtils.elements_individual_vqe_energy_reductions(vqe_runner, elements,
                                                                                 elements_parameters=elements_parameters,
                                                                                 ansatz=ansatz,
                                                                                 ansatz_parameters=ansatz_parameters,
                                                                                 excited_state=excited_state,
                                                                                 global_cache=global_cache)
        elements_results.sort(key=lambda x: x[1].fun)
        if n == 1:
            return min(elements_results, key=lambda x: x[1].fun)
        else:
            return elements_results[:n]  # TODO check


class GradientUtils:

    @staticmethod
    @ray.remote
    def get_excitation_gradient_multithread(excitation, ansatz, ansatz_parameters, q_system, backend, thread_cache=None,
                                            excited_state=0,
                                            n_shots=1024, noise_model=None, coupling_map=None, method='automatic'):

        if backend is QasmBackend:
            if thread_cache is not None:
                raise Exception('thread_cache not supported yet')
            t0 = time.time()
            gradient = backend.ansatz_element_gradient(excitation, ansatz_parameters, ansatz, q_system,
                                                       n_shots=n_shots, noise_model=noise_model,
                                                       coupling_map=coupling_map, method=method)
        else:
            t0 = time.time()
            gradient = backend.ansatz_element_gradient(excitation, ansatz_parameters, ansatz, q_system,
                                                       cache=thread_cache, excited_state=excited_state)

        message = 'Excitation {}. Excitation grad {}. Time {}'.format(excitation.element, gradient, time.time() - t0)
        # TODO check if required
        del thread_cache
        print(message)  # keep this since logging does not work well in multithreading
        return gradient

    # finds energy gradient of <H> w.r.t. to the ansatz_elements variational parameters
    @staticmethod
    def get_ansatz_elements_gradients(elements, q_system, ansatz_parameters=None, ansatz=None,
                                      global_cache=None, backend=backends.QiskitSimBackend, excited_state=0,
                                      n_shots=1024, noise_model=None, coupling_map=None, method='automatic'):

        if ansatz is None:
            ansatz = []
            ansatz_parameters = []

        if backend is QasmBackend:
            if global_cache is not None:
                raise Exception('global_cache not supported for QasmBackend yet')
            if excited_state != 0:
                raise Exception('excited state not supported for QasmBackend yet')

        def get_thread_cache(element):
            if global_cache is not None:
                sparse_statevector = global_cache.get_statevector(ansatz, ansatz_parameters)
                return global_cache.get_grad_thread_cache(element, sparse_statevector)
            else:
                return None

        if config.multithread:
            ray.init(num_cpus=config.ray_options['n_cpus'])
            elements_ray_ids = [
                [
                    element, GradientUtils.get_excitation_gradient_multithread.
                    remote(element, ansatz, ansatz_parameters, q_system, backend,
                           thread_cache=get_thread_cache(element),
                           excited_state=excited_state,
                           n_shots=n_shots, noise_model=noise_model, coupling_map=coupling_map, method=method)
                ]
                for element in elements
            ]
            elements_results = [[element_ray_id[0], ray.get(element_ray_id[1])] for element_ray_id in elements_ray_ids]
            ray.shutdown()
        else:
            if backend is QasmBackend:
                elements_results = [
                    [
                        element,
                        backend.ansatz_element_gradient(ansatz_element=element, var_parameters=ansatz_parameters,
                                                              ansatz=ansatz, q_system=q_system,
                                                              n_shots=n_shots, noise_model=noise_model,
                                                              coupling_map=coupling_map, method=method)]
                    for element in elements
                ]
            else:
                elements_results = [
                    [
                        element,
                        backend.ansatz_element_gradient(ansatz_element=element, var_parameters=ansatz_parameters,
                                                        ansatz=ansatz, q_system=q_system,
                                                        cache=global_cache, excited_state=excited_state)]
                    for element in elements
                ]

        return elements_results

    # returns the n ansatz elements that with largest energy gradients
    @staticmethod
    def get_largest_gradient_elements(elements, q_system, backend=backends.QiskitSimBackend, ansatz_parameters=None,
                                      ansatz=None, n=1, global_cache=None, excited_state=0,
                                      n_shots=1024, noise_model=None, coupling_map=None, method='automatic'):

        elements_results = GradientUtils.get_ansatz_elements_gradients(elements, q_system,
                                                                       ansatz_parameters=ansatz_parameters,
                                                                       ansatz=ansatz, global_cache=global_cache,
                                                                       backend=backend, excited_state=excited_state,
                                                                       n_shots=n_shots, noise_model=noise_model,
                                                                       coupling_map=coupling_map, method=method)
        elements_results.sort(key=lambda x: abs(x[1]))
        return elements_results[-n:]


class DataUtils:
    @staticmethod
    def save_data(data_frame, molecule, time_stamp, ansatz_element_type=None, frozen_els=None, iter_vqe_type='iqeb'):
        filename = '{}_{}_{}_{}_{}.csv'.format(molecule.name, iter_vqe_type, ansatz_element_type, frozen_els,
                                               time_stamp)
        try:
            data_frame.to_csv('../../results/iter_vqe_results/' + filename)
        except FileNotFoundError:
            try:
                data_frame.to_csv('results/iter_vqe_results/' + filename)
            except FileNotFoundError as fnf:
                print(fnf)

    # TODO: make this less ugly and more general
    @staticmethod
    def ansatz_from_data_frame(data_frame, q_system):
        ansatz_elements = []
        for i in range(len(data_frame)):
            element = data_frame.loc[i]['element']
            element_qubits = data_frame.loc[i]['element_qubits']
            if element[0] == 'e' and element[4] == 's':
                ansatz_elements. \
                    append(EffSFExc(ast.literal_eval(element_qubits)[0][0], ast.literal_eval(element_qubits)[1][0],
                                    system_n_qubits=q_system.n_qubits))
            elif element[0] == 'e' and element[4] == 'd':
                ansatz_elements.append(EffDFExc(*ast.literal_eval(element_qubits), system_n_qubits=q_system.n_qubits))
            elif element[0:3] == 's_f':
                ansatz_elements.append(SFExc(*ast.literal_eval(element_qubits), system_n_qubits=q_system.n_qubits))
            elif element[0:3] == 'd_f':
                ansatz_elements.append(DFExc(*ast.literal_eval(element_qubits), system_n_qubits=q_system.n_qubits))
            elif element[0] == 's' and element[2] == 'q':
                # ansatz_elements.append(SQExc(*ast.literal_eval(element_qubits), system_n_qubits=q_system.n_qubits))
                ansatz_elements \
                    .append(SQExc(ast.literal_eval(element_qubits)[0][0], ast.literal_eval(element_qubits)[1][0],
                                  system_n_qubits=q_system.n_qubits))
            elif element[0] == 'd' and element[2] == 'q':
                ansatz_elements.append(DQExc(*ast.literal_eval(element_qubits), system_n_qubits=q_system.n_qubits))
            elif element[:2] == '1j':
                ansatz_elements.append(PauliStringExc(QubitOperator(element), system_n_qubits=q_system.n_qubits))
            elif element[:8] == 'spin_s_f':
                try:
                    ansatz_elements.append(
                        SpinCompEffSFExc(*ast.literal_eval(element_qubits), system_n_qubits=q_system.n_qubits))
                except TypeError:
                    # new format
                    ansatz_elements.append(SpinCompEffSFExc(ast.literal_eval(element_qubits)[0][0],
                                                            ast.literal_eval(element_qubits)[1][0],
                                                            system_n_qubits=q_system.n_qubits))
            elif element[:8] == 'spin_d_f':
                ansatz_elements.append(
                    SpinCompEffDFExc(*ast.literal_eval(element_qubits), system_n_qubits=q_system.n_qubits))
            elif element[:8] == 'spin_s_q':
                ansatz_elements.append(
                    SpinCompSQExc(*ast.literal_eval(element_qubits), system_n_qubits=q_system.n_qubits))
            elif element[:8] == 'spin_d_q':
                ansatz_elements.append(
                    SpinCompDQExc(*ast.literal_eval(element_qubits), system_n_qubits=q_system.n_qubits))
            else:
                print(element, element_qubits)
                raise Exception('Unrecognized ansatz element.')

        var_pars = list(data_frame['var_parameters'])

        return State(ansatz_elements, var_pars, q_system.n_qubits, q_system.n_electrons)

    @staticmethod
    def ansatz_elements_from_data_frame(data_frame, q_system):
        ansatz_elements = []
        for i in range(len(data_frame)):
            element = data_frame.loc[i]['element']
            element_qubits = data_frame.loc[i]['element_qubits']
            if element[0] == 'e' and element[4] == 's':
                ansatz_elements. \
                    append(EffSFExc(ast.literal_eval(element_qubits)[0][0], ast.literal_eval(element_qubits)[1][0],
                                    system_n_qubits=q_system.n_qubits))
            elif element[0] == 'e' and element[4] == 'd':
                ansatz_elements.append(EffDFExc(*ast.literal_eval(element_qubits), system_n_qubits=q_system.n_qubits))
            elif element[0:3] == 's_f':
                ansatz_elements.append(SFExc(*ast.literal_eval(element_qubits), system_n_qubits=q_system.n_qubits))
            elif element[0:3] == 'd_f':
                ansatz_elements.append(DFExc(*ast.literal_eval(element_qubits), system_n_qubits=q_system.n_qubits))
            elif element[0] == 's' and element[2] == 'q':
                # ansatz_elements.append(SQExc(*ast.literal_eval(element_qubits), system_n_qubits=q_system.n_qubits))
                ansatz_elements \
                    .append(SQExc(ast.literal_eval(element_qubits)[0][0], ast.literal_eval(element_qubits)[1][0],
                                  system_n_qubits=q_system.n_qubits))
            elif element[0] == 'd' and element[2] == 'q':
                ansatz_elements.append(DQExc(*ast.literal_eval(element_qubits), system_n_qubits=q_system.n_qubits))
            elif element[:2] == '1j':
                ansatz_elements.append(PauliStringExc(QubitOperator(element), system_n_qubits=q_system.n_qubits))
            elif element[:8] == 'spin_s_f':
                try:
                    ansatz_elements.append(
                        SpinCompEffSFExc(*ast.literal_eval(element_qubits), system_n_qubits=q_system.n_qubits))
                except TypeError:
                    # new format
                    ansatz_elements.append(SpinCompEffSFExc(ast.literal_eval(element_qubits)[0][0],
                                                            ast.literal_eval(element_qubits)[1][0],
                                                            system_n_qubits=q_system.n_qubits))
            elif element[:8] == 'spin_d_f':
                ansatz_elements.append(
                    SpinCompEffDFExc(*ast.literal_eval(element_qubits), system_n_qubits=q_system.n_qubits))
            elif element[:8] == 'spin_s_q':
                ansatz_elements.append(
                    SpinCompSQExc(*ast.literal_eval(element_qubits), system_n_qubits=q_system.n_qubits))
            elif element[:8] == 'spin_d_q':
                ansatz_elements.append(
                    SpinCompDQExc(*ast.literal_eval(element_qubits), system_n_qubits=q_system.n_qubits))
            else:
                print(element, element_qubits)
                raise Exception('Unrecognized ansatz element.')

        return ansatz_elements
