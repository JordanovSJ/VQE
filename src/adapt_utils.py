import ray
from openfermion import get_sparse_operator, QubitOperator
from src import config
from src import backends


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
                 vqe_runner.vqe_run_multithread.remote(self=vqe_runner, ansatz_elements=initial_ansatz + [element],
                                                       initial_var_parameters=initial_var_parameters + [0])]  # TODO this will work only if the ansatz element has 1 var. par.
                for element in ansatz_elements
            ]
            elements_results = [[element_ray_id[0], ray.get(element_ray_id[1])] for element_ray_id in elements_ray_ids]
            ray.shutdown()
        else:
            elements_results = [
                [element, vqe_runner.vqe_run(ansatz_elements=initial_ansatz + [element],
                                             initial_var_parameters=initial_var_parameters)]
                for element in ansatz_elements
            ]

        return elements_results

    # returns the ansatz element that achieves lowest energy (together with the energy value)
    @staticmethod
    def get_most_significant_ansatz_element(vqe_runner, ansatz_elements, initial_var_parameters=None,
                                            initial_ansatz=None, multithread=False):
        elements_results = EnergyAdaptUtils.get_ansatz_elements_vqe_energies(vqe_runner, ansatz_elements,
                                                                             initial_var_parameters=initial_var_parameters,
                                                                             initial_ansatz=initial_ansatz,
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
    @ray.remote
    def get_excitation_energy_gradient_multithread(excitation_element, ansatz_elements, var_parameters, q_system,
                                                   backend, dynamic_commutators=False):

        excitation = excitation_element.excitation
        assert type(excitation) == QubitOperator

        # # TODO make pretier
        if dynamic_commutators:
            commutator_matrix = q_system.get_commutator_matrix(excitation_element)
        else:
            commutator_matrix = \
                get_sparse_operator(q_system.jw_qubit_ham*excitation - excitation*q_system.jw_qubit_ham).todense()

        gradient = backend.get_expectation_value(q_system, ansatz_elements, var_parameters,
                                                 operator_matrix=commutator_matrix)[0]

        message = 'Excitation {}. Excitation grad {}'.format(excitation_element.element, gradient)
        print(message)
        return gradient

    @staticmethod
    def get_excitation_energy_gradient(excitation_element, ansatz_elements, var_parameters, q_system, backend,
                                       dynamic_commutators=False):

        excitation = excitation_element.excitation
        assert type(excitation) == QubitOperator

        if dynamic_commutators:
            commutator_matrix = q_system.get_commutator_matrix(excitation_element)
        else:
            commutator_matrix = \
                get_sparse_operator(q_system.jw_qubit_ham * excitation - excitation * q_system.jw_qubit_ham).todense()

        gradient = backend.get_expectation_value(q_system, ansatz_elements, var_parameters,
                                                 operator_matrix=commutator_matrix)[0]

        message = 'Excitation {}. Excitation grad {}'.format(excitation_element.element, gradient)
        print(message)

        return gradient

    # finds the VQE energy contribution of a single ansatz element added to (optionally) an initial ansatz
    @staticmethod
    def ansatz_elements_gradients(ansatz_elements, q_system, backend, initial_var_parameters=None,
                                  initial_ansatz=None, multithread=False, dynamic_commutators=False):

        if initial_ansatz is None:
            initial_ansatz = []
        if multithread:
            ray.init(num_cpus=config.multithread['n_cpus'])
            elements_ray_ids = [
                [element,
                 GradAdaptUtils.get_excitation_energy_gradient_multithread.remote(element, initial_ansatz,
                                                                                  initial_var_parameters, q_system,
                                                                                  backend, dynamic_commutators)]
                for element in ansatz_elements
            ]
            elements_results = [[element_ray_id[0], ray.get(element_ray_id[1])] for element_ray_id in
                                elements_ray_ids]
            ray.shutdown()
        else:
            elements_results = [
                [element, GradAdaptUtils.get_excitation_energy_gradient(element, initial_ansatz, initial_var_parameters,
                                                                        q_system, backend, dynamic_commutators)]
                for element in ansatz_elements
            ]

        return elements_results

    # returns the ansatz element that achieves lowest energy (together with the energy value)
    @staticmethod
    def most_significant_ansatz_elements(ansatz_elements, q_system, backend, var_parameters=None, ansatz=None, n=1,
                                         multithread=False, dynamic_commutators=False):

        elements_results = GradAdaptUtils.ansatz_elements_gradients(ansatz_elements, q_system, backend,
                                                                    initial_var_parameters=var_parameters,
                                                                    initial_ansatz=ansatz,
                                                                    multithread=multithread,
                                                                    dynamic_commutators=dynamic_commutators)
        elements_results.sort(key=lambda x: abs(x[1]))
        return elements_results[-n:]
