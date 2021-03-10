import pandas as pd

from src.molecules.molecules import LiH, H2, H4
from scripts.zhenghao.noisy_backends import QasmBackend
from scripts.zhenghao.test_utils import *
from src.ansatz_element_sets import UCCSD
from src.backends import QiskitSimBackend
from src.utils import *
from src.iter_vqe_utils import DataUtils

from qiskit import IBMQ
from qiskit.providers.aer.noise import NoiseModel

import time
import numpy as np

r_list = [1.546, 2, 2.5, 3, 3.5]
molecule = H4()
n_qubits = molecule.n_qubits
n_electrons = molecule.n_electrons

n_shots = 1e4
method = 'automatic'
prob_1 = 0

# <<<<<<<<<< LOGGING >>>>>>>>>>>>.
LogUtils.log_config()

logging.info('{}, noisy ham_expectation_value for ansatz constructed from iqeb'
             .format(molecule.name, r))

logging.info('n_qubits = {}, n_electrons = {}'.format(n_qubits, n_electrons))
time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

logging.info('n_shots={}, method={})').format(n_shots, method)

# <<<<<<<<<< INITIALISE DATAFRAME TO COLLECT RESULTS >>>>>>>>>>>>.
results_df = pd.DataFrame(columns=['r', 'ref_E', 'cnot_count', 'u1_count',
                                   'cnot_depth', 'u1_depth',
                                   'error=1e-1', 'time_1',
                                   'error=1e-2', 'time_2',
                                   'error=1e-3', 'time_3'])
filename = '../../results/zhenghao_testing/' \
                '{}_dissociation_iqeb_ansatz_noisy_compare_{}.csv'\
    .format(molecule.name, time_stamp)


idx = 0
for r in r_list:
    ref_result, cnot_count, u1_count, cnot_depth, u1_depth = \
        TestUtils.h4_depth_count_from_csv(r)

    prob_2 = 1e-1
    prob_meas=1e-1

    logging.info('')
    logging.info('prob_2={}, prob_meas={}').format(prob_2, prob_meas)

    noise_model = NoiseUtils.gate_and_measure_noise(prob_1=prob_1, prob2=prob_2,
                                                    prob_meas=prob_meas)
    expectation_value_1, time_1 = \
        TestUtils.h4_noisy_iqeb_ansatz_evaluation(r=r, noise_model=noise_model,
                                                  n_shots=n_shots, method='method')
    logging.info('QasmSimulator for expectation_value={}, time_used={}')\
        .format(expectation_value_1, time_1)

    prob_2 = 1e-2
    prob_meas = 1e-2

    noise_model = NoiseUtils.gate_and_measure_noise(prob_1=prob_1, prob2=prob_2,
                                                    prob_meas=prob_meas)
    expectation_value_2, time_2 = \
        TestUtils.h4_noisy_iqeb_ansatz_evaluation(r=r, noise_model=noise_model,
                                                  n_shots=n_shots, method='method')

    prob_2 = 1e-3
    prob_meas = 1e-3

    noise_model = NoiseUtils.gate_and_measure_noise(prob_1=prob_1, prob2=prob_2,
                                                    prob_meas=prob_meas)
    expectation_value_3, time_3 = \
        TestUtils.h4_noisy_iqeb_ansatz_evaluation(r=r, noise_model=noise_model,
                                                  n_shots=n_shots, method='method')

    results_df.loc[idx] = {
        'r': r, 'ref_E': ref_result, 'cnot_count': cnot_count, 'u1_count': u1_count,
        'cnot_depth': cnot_depth, 'u1_depth': u1_depth,
        'error=1e-1': expectation_value_1, 'time_1': time_1,
        'error=1e-2': expectation_value_2, 'time_2': time_2,
        'error=1e-3': expectation_value_3, 'time_3': time_3
    }
    results_df.to_csv(filename)

    idx+=1



