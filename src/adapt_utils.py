import ray
from openfermion import get_sparse_operator, QubitOperator
from src import config
from src import backends
import time


class EnergyAdaptUtils:
    # finds the VQE energy contribution of a single ansatz element added to (optionally) an initial ansatz
    @staticmethod
    def get_ansatz_elements_vqe_energies(vqe_runner, ansatz_elements, initial_var_parameters=None,
                                         initial_ansatz=None, multithread=False):
        if initial_ansatz is None:
            initial_ansatz = []
        if multithread:
            ray.init(num_cpus=config.multithread['n_cpus'])
            elements_ray_ids = [
                [element,
                 vqe_runner.vqe_run_multithread.remote(self=vqe_runner, ansatz=initial_ansatz + [element],
                                                       initial_var_parameters=initial_var_parameters + [0])]
                # TODO this will work only if the ansatz element has 1 var. par.
                for element in ansatz_elements
            ]
            elements_results = [[element_ray_id[0], ray.get(element_ray_id[1])] for element_ray_id in elements_ray_ids]
            ray.shutdown()
        else:
            elements_results = [
                [element, vqe_runner.vqe_run(ansatz=initial_ansatz + [element],
                                             initial_var_parameters=initial_var_parameters+[0])]
                for element in ansatz_elements
            ]

        return elements_results

    # returns the ansatz element that achieves lowest energy (together with the energy value)
    @staticmethod
    def get_most_significant_ansatz_element(vqe_runner, ansatz_elements, initial_var_parameters=None,
                                            ansatz=None, multithread=False):
        elements_results = EnergyAdaptUtils.get_ansatz_elements_vqe_energies(vqe_runner, ansatz_elements,
                                                                             initial_var_parameters=initial_var_parameters,
                                                                             initial_ansatz=ansatz,
                                                                             multithread=multithread)
        return min(elements_results, key=lambda x: x[1].fun)

    # get ansatz elements that contribute to energy decrease below(above) some threshold value
    @staticmethod
    def get_ansatz_elements_below_threshold(vqe_runner, ansatz_elements, threshold, initial_var_parameters=[],
                                            initial_ansatz=[], multithread=False):
        elements_results = EnergyAdaptUtils.get_ansatz_elements_vqe_energies(vqe_runner, ansatz_elements,
                                                                             initial_var_parameters=initial_var_parameters,
                                                                             initial_ansatz=initial_ansatz,
                                                                             multithread=multithread)
        return [element_result for element_result in elements_results if element_result[1].fun <= threshold]


class GradAdaptUtils:

    @staticmethod
    def compute_commutators(qubit_ham, ansatz_elements, n_system_qubits, multithread=False):
        commutators = {}
        if multithread:
            ray.init(num_cpus=config.multithread['n_cpus'])
            elements_ray_ids = [
                [
                    element, GradAdaptUtils.get_commutator_matrix_multithread.
                    remote(QubitOperator(str(element.excitation_generator)), qubit_ham=QubitOperator(str(qubit_ham)), n_qubits=n_system_qubits) # passing copies UGLY
                ]
                for element in ansatz_elements
            ]
            for element_ray_id in elements_ray_ids:
                key = str(element_ray_id[0].excitation_generator)
                commutators[key] = ray.get(element_ray_id[1])

            del elements_ray_ids
            ray.shutdown()
        else:
            for i, ansatz_element in enumerate(ansatz_elements):
                element_excitation = ansatz_element.excitation_generator
                key = str(element_excitation)
                print('Calculated commutator ', key)
                commutator = qubit_ham * element_excitation - element_excitation * qubit_ham
                commutator_matrix = get_sparse_operator(commutator, n_qubits=n_system_qubits)
                commutators[key] = commutator_matrix

        return commutators

    @staticmethod
    @ray.remote
    def get_commutator_matrix_multithread(excitation, qubit_ham, n_qubits):
        t0 = time.time()
        commutator = qubit_ham * excitation - excitation * qubit_ham
        commutator_sparse_matrix = get_sparse_operator(commutator,  n_qubits=n_qubits)
        print('Calculated commutator ', str(excitation), 'time ', time.time() - t0)
        del commutator
        del t0
        del excitation
        del qubit_ham
        return commutator_sparse_matrix

    @staticmethod
    @ray.remote
    def get_excitation_energy_gradient_multithread(excitation_element, ansatz, var_parameters, backend,
                                                   commutator_sparse_matrix=None):

        t0 = time.time()
        gradient = backend.get_excitation_gradient(excitation_element, ansatz, var_parameters,
                                                   commutator_sparse_matrix=commutator_sparse_matrix,
                                                   update_statevector=False)  # experiment with true

        message = 'Excitation {}. Excitation grad {}. Time {}'.format(excitation_element.element, gradient,
                                                                      time.time() - t0)
        # TODO check if required
        del commutator_sparse_matrix
        print(message)
        return gradient

    # @staticmethod
    # def get_excitation_energy_gradient(excitation_element, ansatz, var_parameters, backend, commutator_sparse_matrix=None):
    #
    #     t0 = time.time()
    #     gradient = backend.get_excitation_gradient(excitation_element, ansatz, var_parameters,
    #                                                commutator_sparse_matrix=commutator_sparse_matrix,
    #                                                update_statevector=False)  # experiment with true
    #
    #     message = 'Excitation {}. Excitation grad {}. Time {}'.format(excitation_element.element, gradient,
    #                                                                   time.time() - t0)
    #     print(message)
    #
    #     return gradient

    # finds the VQE energy contribution of a single ansatz element added to (optionally) an initial ansatz
    @staticmethod
    def ansatz_elements_gradients(ansatz_elements, q_system, initial_var_parameters=None, initial_ansatz=None,
                                  multithread=False, dynamic_commutators=None, backend_type=backends.QiskitSim):

        if initial_ansatz is None:
            initial_ansatz = []
            initial_var_parameters = []

        backend = backend_type(q_system, store_H_sparse_matrx=True)
        # initialize the statevector once, and use it to calculate all gradients
        backend.get_updated_statevector(initial_ansatz, initial_var_parameters)

        # if do_precompute_statevector:
        #     precomputed_statevector = backend.statevector_from_ansatz(initial_ansatz, initial_var_parameters,
        #                                                               q_system.n_orbitals, q_system.n_electrons)[0]
        # else:
        #     precomputed_statevector = None

        def dynamic_commutator_matrix(element):
            if dynamic_commutators is None:
                return None
            else:
                try:
                    return dynamic_commutators[str(element.excitation_generator)].copy()
                except KeyError:
                    t0 = time.time()
                    commutator = q_system.jw_qubit_ham * element.excitation_generator - element.excitation_generator * q_system.jw_qubit_ham
                    commutator_matrix = get_sparse_operator(commutator)
                    dynamic_commutators[str(element.excitation_generator)] = commutator_matrix
                    print('Calculating commutator for ', element.excitation_generator, 'time ', time.time() - t0)
                    return commutator_matrix.copy()

        if multithread:
            ray.init(num_cpus=config.multithread['n_cpus'])
            elements_ray_ids = [
                [
                    element, GradAdaptUtils.get_excitation_energy_gradient_multithread.
                    remote(element, initial_ansatz, initial_var_parameters, backend,
                           commutator_sparse_matrix=dynamic_commutator_matrix(element))  # precomputed_statevector.copy() ??
                 ]
                for element in ansatz_elements
            ]
            elements_results = [[element_ray_id[0], ray.get(element_ray_id[1])] for element_ray_id in
                                elements_ray_ids]
            ray.shutdown()
        else:
            elements_results = [
                [
                    element,
                    backend.get_excitation_gradient(element, initial_ansatz, initial_var_parameters,
                                                    commutator_sparse_matrix=dynamic_commutator_matrix(element),
                                                    update_statevector=True)]
                for element in ansatz_elements
            ]
        return elements_results

    # returns the ansatz element that achieves lowest energy (together with the energy value)
    @staticmethod
    def most_significant_ansatz_elements(ansatz_elements, q_system, backend_type=backends.QiskitSim, var_parameters=None
                                         , ansatz=None, n=1, multithread=False, dynamic_commutators=None):

        elements_results = GradAdaptUtils.ansatz_elements_gradients(ansatz_elements, q_system,
                                                                    initial_var_parameters=var_parameters,
                                                                    initial_ansatz=ansatz,
                                                                    multithread=multithread,
                                                                    dynamic_commutators=dynamic_commutators,
                                                                    backend_type=backend_type)
        elements_results.sort(key=lambda x: abs(x[1]))
        return elements_results[-n:]
