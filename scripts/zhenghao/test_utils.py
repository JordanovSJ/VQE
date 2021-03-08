from qiskit import IBMQ
from qiskit.providers.aer.noise import NoiseModel, errors, pauli_error

from src.utils import *
from src.iter_vqe_utils import *
from scripts.zhenghao.noisy_backends import QasmBackend
from src.molecules.molecules import H4

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

    @staticmethod
    def h4_depth_count_from_csv(r):
        molecule = H4(r=r)
        data_frame = TestUtils.df_from_csv(molecule, r)

        reference_results = data_frame['E']
        ref_size = len(reference_results)
        ref_result = reference_results[ref_size - 1]

        cnot_count = data_frame['cnot_count'][ref_size - 1]
        u1_count = data_frame['u1_count'][ref_size - 1]
        cnot_depth = data_frame['cnot_depth'][ref_size - 1]
        u1_depth = data_frame['u1_depth'][ref_size - 1]

        return ref_result, cnot_count, u1_count, cnot_depth, u1_depth

    @staticmethod
    def h4_noisy_iqeb_ansatz_evaluation(r, noise_model, n_shots, method):
        molecule = H4(r=r)
        data_frame = TestUtils.df_from_csv(molecule, r)
        ansatz_state = DataUtils.ansatz_from_data_frame(data_frame, molecule)

        ansatz = ansatz_state.ansatz_elements
        var_pars = ansatz_state.parameters

        t0 = time.time()
        expectation_value = QasmBackend.ham_expectation_value(var_pars, ansatz,
                                                              molecule, n_shots=n_shots,
                                                              noise_model=noise_model,
                                                              method=method,
                                                              built_in_Pauli=True)
        t1 = time.time()

        return expectation_value, t1 - t0
