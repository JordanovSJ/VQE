from openfermion import get_sparse_operator, QubitOperator
from src import config
from src import backends
from src.ansatz_elements import *
from src.utils import QasmUtils
from src.ansatze import Ansatz
from src.backends import QiskitSim

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
        qasm = backends.QiskitSim.qasm_from_ansatz(ansatz, var_parameters)
        return QasmUtils.gate_count_from_qasm(qasm, n_qubits)


class EnergyUtils:
    # calculate the full (optimizing all parameters) VQE energy reductions for a set of ansatz elements
    @staticmethod
    def elements_full_vqe_energy_reductions(vqe_runner, ansatz_elements, var_parameters=None, ansatz=None,
                                            global_cache=None, excited_state=0):

        if ansatz is None:
            ansatz = []

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
                                                       initial_var_parameters=var_parameters + [0],
                                                       cache=get_thread_cache(), excited_state=excited_state)]
                # TODO this will work only if the ansatz element has 1 var. par.
                for element in ansatz_elements
            ]
            elements_results = [[element_ray_id[0], ray.get(element_ray_id[1])] for element_ray_id in elements_ray_ids]
            ray.shutdown()
        else:
            elements_results = [
                [element, vqe_runner.vqe_run(ansatz=ansatz + [element], excited_state=excited_state,
                                             initial_var_parameters=var_parameters + [0], cache=global_cache)]
                for element in ansatz_elements
            ]

        return elements_results

    # returns the ansatz element that achieves the largest full (optimizing all parameters) VQE energy reduction
    @staticmethod
    def largest_full_vqe_energy_reduction_element(vqe_runner, ansatz_elements, ansatz=None, var_parameters=None,
                                                  global_cache=None, excited_state=0):
        elements_results = EnergyUtils.elements_full_vqe_energy_reductions(vqe_runner, ansatz_elements,
                                                                           var_parameters=var_parameters,
                                                                           ansatz=ansatz, excited_state=excited_state,
                                                                           global_cache=global_cache)
        return min(elements_results, key=lambda x: x[1].fun)

    # # NOT used
    # # get ansatz elements that contribute to energy reduction below(above) some threshold value
    # @staticmethod
    # def elements_below_full_vqe_energy_reduction_threshold(vqe_runner, ansatz_elements, threshold, ansatz=[],
    #                                                        var_parameters=[], multithread=False, excited_state=0):
    #     elements_results = EnergyUtils.elements_full_vqe_energy_reductions(vqe_runner, ansatz_elements,
    #                                                                        var_parameters=var_parameters,
    #                                                                        ansatz=ansatz, multithread=multithread,
    #                                                                        excited_state=excited_state)
    #     return [element_result for element_result in elements_results if element_result[1].fun <= threshold]

    # calculate the full (optimizing all parameters) VQE energy reductions for a set of ansatz elements
    @staticmethod
    def elements_individual_vqe_energy_reductions(vqe_runner, ansatz_elements, ansatz=None, var_parameters=None,
                                                  excited_state=0, global_cache=None):

        if ansatz is None:
            ansatz = []
            var_parameters = []

        ansatz_qasm = QasmUtils.hf_state(vqe_runner.q_system.n_electrons)
        ansatz_qasm += vqe_runner.backend.qasm_from_ansatz(ansatz, var_parameters)

        if config.multithread:
            ray.init(num_cpus=config.ray_options['n_cpus'])
            # TODO remove this if statement
            if vqe_runner.backend == QiskitSim and global_cache is not None:
                # TODO move in cache class?
                init_statevector = global_cache.update_statevector(ansatz, var_parameters)
                init_sparse_statevector = scipy.sparse.csr_matrix(init_statevector).transpose().conj()
                elements_ray_ids = [
                    [element,
                     vqe_runner.vqe_run_single_parameter_multithread.
                        remote(self=vqe_runner, ansatz_element=element,
                               cache=global_cache.single_par_vqe_thread_cache(element, init_statevector=init_sparse_statevector))
                     ]
                    for element in ansatz_elements
                ]
            else:
                elements_ray_ids = [
                    [element,
                     vqe_runner.vqe_run_multithread.remote(self=vqe_runner, ansatz=[element], init_state_qasm=ansatz_qasm,
                                                           initial_var_parameters=[0], excited_state=excited_state)
                     ]
                    # TODO this will work only if the ansatz element has 1 var. par.
                    for element in ansatz_elements
                ]
            elements_results = [[element_ray_id[0], ray.get(element_ray_id[1])] for element_ray_id in
                                elements_ray_ids]
            ray.shutdown()
        else:
            elements_results = [
                [element, vqe_runner.vqe_run(ansatz=[element], initial_var_parameters=[0], excited_state=excited_state,
                                             init_state_qasm=ansatz_qasm)
                 ]
                for element in ansatz_elements
            ]

        return elements_results

    # returns the ansatz element that achieves the largest full (optimizing all parameters) VQE energy reduction
    @staticmethod
    def largest_individual_element_vqe_energy_reduction(ansatz_elements, vqe_runner, ansatz=None, var_parameters=None,
                                                        global_cache=None, excited_state=0):
        elements_results = EnergyUtils.elements_individual_vqe_energy_reductions(vqe_runner, ansatz_elements,
                                                                                 ansatz=ansatz, var_parameters=var_parameters,
                                                                                 excited_state=excited_state,
                                                                                 global_cache=global_cache)
        return min(elements_results, key=lambda x: x[1].fun)


class GradUtils:

    @staticmethod
    @ray.remote
    def get_excitation_gradient_multithread(excitation, ansatz, var_parameters, q_system, backend, thread_cache=None,
                                            commutator_sparse_matrix=None, excited_state=0):
        t0 = time.time()
        gradient = backend.excitation_gradient(excitation, ansatz, var_parameters, q_system, cache=thread_cache,
                                               excited_state=excited_state)

        message = 'Excitation {}. Excitation grad {}. Time {}'.format(excitation.element, gradient, time.time() - t0)
        # TODO check if required
        del thread_cache
        print(message)  # keep this since logging does not work well in multithreading
        return gradient

    # finds energy gradient of <H> w.r.t. to the ansatz_elements variational parameters
    @staticmethod
    def get_ansatz_elements_gradients(ansatz_elements, q_system, var_parameters=None, ansatz=None,
                                      global_cache=None, backend=backends.QiskitSim, excited_state=0):

        if ansatz is None:
            ansatz = []
            var_parameters = []

        def get_thread_cache(element):
            if global_cache is not None:
                statevector = global_cache.update_statevector(ansatz, var_parameters)
                sparse_statevector = scipy.sparse.csr_matrix(statevector).transpose().conj()
                return global_cache.get_grad_thread_cache(element, sparse_statevector)
            else:
                return None

        if config.multithread:
            ray.init(num_cpus=config.ray_options['n_cpus'])
            elements_ray_ids = [
                [
                    element, GradUtils.get_excitation_gradient_multithread.
                    remote(element, ansatz, var_parameters, q_system, backend, thread_cache=get_thread_cache(element),
                           excited_state=excited_state)
                 ]
                for element in ansatz_elements
            ]
            elements_results = [[element_ray_id[0], ray.get(element_ray_id[1])] for element_ray_id in elements_ray_ids]
            ray.shutdown()
        else:
            elements_results = [
                [
                    element, backend.excitation_gradient(element, ansatz, var_parameters, q_system,
                                                         cache=get_thread_cache(element), excited_state=excited_state)]
                for element in ansatz_elements
            ]
        return elements_results

    # returns the n ansatz elements that with largest energy gradients
    @staticmethod
    def get_largest_gradient_ansatz_elements(ansatz_elements, q_system, backend=backends.QiskitSim, var_parameters=None,
                                             ansatz=None, n=1, global_cache=None, excited_state=0):

        elements_results = GradUtils.get_ansatz_elements_gradients(ansatz_elements, q_system,
                                                                   var_parameters=var_parameters,
                                                                   ansatz=ansatz, global_cache=global_cache,
                                                                   backend=backend, excited_state=excited_state)
        elements_results.sort(key=lambda x: abs(x[1]))
        return elements_results[-n:]


class DataUtils:
    @staticmethod
    def save_data(data_frame, molecule, time_stamp, ansatz_element_type=None, frozen_els=None, iter_vqe_type='iqeb'):
        filename = '{}_{}_{}_{}_{}.csv'.format(molecule.name, iter_vqe_type, ansatz_element_type, frozen_els, time_stamp)
        try:
            data_frame.to_csv('../../results/iter_vqe_results/'+filename)
        except FileNotFoundError:
            try:
                data_frame.to_csv('results/iter_vqe_results/'+filename)
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
                ansatz_elements.append(EffSFExc(*ast.literal_eval(element_qubits), system_n_qubits=q_system.n_qubits))
            elif element[0] == 'e' and element[4] == 'd':
                ansatz_elements.append(EffDFExc(*ast.literal_eval(element_qubits), system_n_qubits=q_system.n_qubits))
            elif element[0] == 's' and element[2] == 'q':
                ansatz_elements.append(SQExc(*ast.literal_eval(element_qubits), system_n_qubits=q_system.n_qubits))
            elif element[0] == 'd' and element[2] == 'q':
                ansatz_elements.append(DQExc(*ast.literal_eval(element_qubits), system_n_qubits=q_system.n_qubits))
            elif element[:2] == '1j':
                ansatz_elements.append(PauliStringExc(QubitOperator(element), system_n_qubits=q_system.n_qubits))
            elif element[:8] == 'spin_s_f':
                ansatz_elements.append(SpinCompSFExc(*ast.literal_eval(element_qubits), system_n_qubits=q_system.n_qubits))
            elif element[:8] == 'spin_d_f':
                ansatz_elements.append(SpinCompDFExc(*ast.literal_eval(element_qubits), system_n_qubits=q_system.n_qubits))
            elif element[:8] == 'spin_s_q':
                ansatz_elements.append(SpinCompSQExc(*ast.literal_eval(element_qubits), system_n_qubits=q_system.n_qubits))
            elif element[:8] == 'spin_d_q':
                ansatz_elements.append(SpinCompDQExc(*ast.literal_eval(element_qubits), system_n_qubits=q_system.n_qubits))
            else:
                print(element, element_qubits)
                raise Exception('Unrecognized ansatz element.')

        var_pars = list(data_frame['var_parameters'])

        return Ansatz(ansatz_elements, var_pars, q_system.n_qubits, q_system.n_electrons)

