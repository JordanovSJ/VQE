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
                                                       initial_var_parameters=initial_var_parameters)]
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
    def get_ansatz_elements_below_threshold(vqe_runner, ansatz_elements, threshold, initial_var_parameters=None,
                                            initial_ansatz=None, multithread=False):
        elements_results = EnergyAdaptUtils.get_ansatz_elements_vqe_energies(vqe_runner, ansatz_elements,
                                                                             initial_var_parameters=initial_var_parameters,
                                                                             initial_ansatz=initial_ansatz,
                                                                             multithread=multithread)
        return [element_result for element_result in elements_results if element_result[1].fun <= threshold]


class GradAdaptUtils:

    @staticmethod
    @ray.remote
    def get_excitation_energy_gradient_multithread(excitation_element, ansatz_elements, var_parameters, qubit_hamiltonian,
                                                   n_qubits, n_electrons, backend, initial_statevector_qasm=None):

        excitation = excitation_element.excitation
        assert type(excitation) == QubitOperator
        assert type(qubit_hamiltonian) == QubitOperator

        # TODO consider not doing this if commutator matrix is supplied
        commutator = qubit_hamiltonian*excitation - excitation*qubit_hamiltonian

        gradient = backend.get_expectation_value(commutator, ansatz_elements, var_parameters, n_qubits, n_electrons,
                                                 initial_statevector_qasm=initial_statevector_qasm)[0]

        message = 'Excitation {}. Excitation grad {}'.format(excitation_element.element, gradient)
        print(message)
        return gradient

    @staticmethod
    def get_excitation_energy_gradient(excitation_element, ansatz_elements, var_parameters, qubit_hamiltonian, n_qubits,
                                       n_electrons, backend, initial_statevector_qasm=None):

        excitation = excitation_element.excitation
        assert type(excitation) == QubitOperator
        assert type(qubit_hamiltonian) == QubitOperator

        commutator = qubit_hamiltonian * excitation - excitation * qubit_hamiltonian

        gradient = backend.get_expectation_value(commutator, ansatz_elements, var_parameters, n_qubits, n_electrons,
                                                 initial_statevector_qasm=initial_statevector_qasm)[0]

        message = 'Excitation {}. Excitation grad {}'.format(excitation_element.element, gradient)
        print(message)

        return gradient

    # finds the VQE energy contribution of a single ansatz element added to (optionally) an initial ansatz
    @staticmethod
    def get_ansatz_elements_gradients(ansatz_elements, qubit_ham, n_qubits, n_electrons, backend, initial_var_parameters=None,
                                      initial_ansatz=None, multithread=False):

        if initial_ansatz is None:
            initial_ansatz = []
        if multithread:
            ray.init(num_cpus=config.multithread['n_cpus'])
            elements_ray_ids = [
                [element,
                 GradAdaptUtils.get_excitation_energy_gradient_multithread.remote(element, initial_ansatz,
                                                                                  initial_var_parameters,
                                                                                  qubit_ham, n_qubits, n_electrons, backend)]
                for element in ansatz_elements
            ]
            elements_results = [[element_ray_id[0], ray.get(element_ray_id[1])] for element_ray_id in
                                elements_ray_ids]
            ray.shutdown()
        else:
            elements_results = [
                [element, GradAdaptUtils.get_excitation_energy_gradient(element, initial_ansatz,
                                                                        initial_var_parameters,
                                                                        qubit_ham, n_qubits, n_electrons, backend)]
                for element in ansatz_elements
            ]

        return elements_results

    # returns the ansatz element that achieves lowest energy (together with the energy value)
    @staticmethod
    def get_most_significant_ansatz_element(ansatz_elements, qubit_ham, n_qubits, n_electrons, backend,
                                            initial_var_parameters=None, initial_ansatz=None, multithread=False):

        elements_results = GradAdaptUtils.get_ansatz_elements_gradients(ansatz_elements, qubit_ham, n_qubits, n_electrons,
                                                                        backend, initial_var_parameters=initial_var_parameters,
                                                                        initial_ansatz=initial_ansatz, multithread=multithread)
        return max(elements_results, key=lambda x: abs(x[1]))
