from qiskit import IBMQ
from qiskit.providers.aer.noise import NoiseModel, errors, pauli_error

from src.utils import *
from src.iter_vqe_utils import *
from scripts.zhenghao.noisy_backends import QasmBackend

import time
from functools import partial
import pandas as pd


class NoiseUtils:

    @staticmethod
    def gate_and_measure_noise(prob_1, prob_2, prob_meas):
        # Create empty noise model
        noise_model = NoiseModel()

        # Depolarizing error for single and two qubit gates
        error_1 = errors.depolarizing_error(prob_1, num_qubits=1)
        error_2 = errors.depolarizing_error(prob_2, num_qubits=2)

        # Measurement error: for a prob of p_meas the measurement of 1 gives 0 and 0 gives 1,
        # i.e. a bit flip on measurement gate
        error_meas = pauli_error([('X', prob_meas), ('I', 1 - prob_meas)])

        # Add the errors to all qubits
        noise_model.add_all_qubit_quantum_error(error_1, ['sx', 'x'])
        noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
        noise_model.add_all_qubit_quantum_error(error_meas, ['measure'])

        return noise_model

    @staticmethod
    def device_noise(device_name: str):
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q')
        backend = provider.get_backend(device_name)
        noise_model = NoiseModel.from_backend(backend)
        coupling_map = backend.configuration().coupling_map

        return noise_model, coupling_map

# STILL UNDER CONSTRUCTION
class TestUtils:

    @staticmethod
    def df_from_csv(molecule, r):
        df = pd.read_csv('../../results/iter_vqe_results/{}_iqeb_q_exc_r={}_22-Feb-2021.csv'
                         .format(molecule.name, r))
        return df

    # STILL UNDER CONSTRUCTION
    @staticmethod
    def iqeb_result_noisy_compare(molecule, r, threshold, noise_model=None,
                                  coupling_map=None, methods=None,
                                  n_shots=1024):
        if methods is None:
            methods = ['automatic']

        # READ CSV
        data_frame = TestUtils.df_from_csv(molecule, r)
        reference_results = data_frame['E']
        dE_list = data_frame['dE']
        iteration_number = data_frame['n']
        ansatz_state = DataUtils.ansatz_from_data_frame(data_frame, molecule)

        # ANSATZ
        ansatz = ansatz_state.ansatz_elements
        var_parameters = ansatz_state.parameters

        iter_index = 0

        while dE_list[iter_index] > threshold:
            ansatz_index = iter_index + 1
            noiseless_result = reference_results[iter_index]
            iter_n = iteration_number[iter_index]

            message = 'Running QasmBackend.ham_expectation_value for {} molecule, ' \
                      'first {} ansatz elements, noiseless value={}, {} shots' \
                .format(molecule.name, ansatz_index, noiseless_result, n_shots)
            logging.info(message)

            ham_expectation_value = partial(QasmBackend.ham_expectation_value,
                                            var_parameters=var_parameters[0:ansatz_index],
                                            ansatz=ansatz[0:ansatz_index],
                                            q_system=molecule, n_shots=n_shots,
                                            noise_model=noise_model, coupling_map=coupling_map)

            for method_str in methods:
                t0 = time.time()
                exp_value = ham_expectation_value(method=method_str)
                t1 = time.time()

            iter_index += 1
