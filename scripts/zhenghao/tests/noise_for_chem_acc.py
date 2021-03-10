from src.molecules.molecules import H4
from scripts.zhenghao.test_utils import NoiseUtils, TestUtils
from src.iter_vqe_utils import DataUtils
from src.utils import *

from openfermion import QubitOperator

import pandas as pd
from functools import partial
import time

# <<<<<<<<<<<< MOLECULE >>>>>>>>>>>>>>>>>
r = 1
molecule = H4(r=r)
n_qubits = molecule.n_qubits
n_electrons = molecule.n_electrons

# <<<<<<<<<< LOGGING >>>>>>>>>>>>.
LogUtils.log_config()
logging.info('{}, test noise levels for 5 hamiltonian terms'.format(molecule.name))
logging.info('n_qubits = {}, n_electrons = {}'.format(n_qubits, n_electrons))
time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

# <<<<<<<<<<<< TEST HAMILTONIAN >>>>>>>>>>>>>>>>>
# Take 5 random terms out of H4 hamiltonian
ham = QubitOperator('Y0 Z1 Y2 X4 Z5 X6', 0.012709132884803409)
ham += QubitOperator('Y0 Z1 Z2 X3 X4 Z5 Z6 Y7', 0.02318657944505642)
ham += QubitOperator('ZO Z1', 0.14266723882946233)
ham += QubitOperator('Y3 Z4 Z5 Y7', 0.028916993734486442)
ham += QubitOperator('Y1 Y2 X3 X4', -0.026485313086317092)

message = 'Hamiltonian is {}'.format(ham)
logging.info(message)

# <<<<<<<<<<<< Noise Model >>>>>>>>>>>>>>>>>
# Noise model
prob_1 = 0  # Single qubit gate depolarizing error prob
time_single_gate = 0  # Gate time for single qubit gate
time_cx = 900  # Gate time for cx gate
time_meas = 900
t1 = 50e3  # T1 in nanoseconds
t2 = 50e3  # T2 in nanoseconds

message = 'prob_1 = {}, time_single_gate={}, time_cx={}, time_meas={}, t1={}, t2={}' \
    .format(prob_1, time_single_gate, time_cx, time_meas, t1, t2)
logging.info(message)

unified_noise = partial(NoiseUtils.unified_noise, prob_1=prob_1,
                        time_single_gate=time_single_gate,
                        time_cx=time_cx, time_measure=time_meas,
                        t1=t1, t2=t2)

noise_model_list = []
prob_2_list = [1e-2, 1e-3, 1e-4, 1e-5, 0]
prob_meas_list = [1e-2, 1e-3, 1e-4, 1e-5, 0]
for prob_2, prob_meas in zip(prob_2_list, prob_meas_list):
    noise_model_list.append(unified_noise(prob_2, prob_meas))

message = 'Noise models generated for prob_2 = {}, prob_meas = {}'.format(prob_2_list, prob_meas_list)
logging.info(message)

# <<<<<<<<<<<< READING CSV FILES >>>>>>>>>>>>>>>>>
df_q_exc = pd.read_csv('../../../results/iter_vqe_results/'
                       'H4_adapt_vqe_q_exc_r=1_08-Mar-2021.csv')
ansatz_state_q_exc = DataUtils.ansatz_from_data_frame(df_q_exc, molecule)
ansatz_q_exc = ansatz_state_q_exc.ansatz_elements
var_pars_q_exc = ansatz_state_q_exc.parameters

df_f_exc = pd.read_csv('../../../results/iter_vqe_results/'
                       'H4_adapt_vqe_eff_f_exc_r=1_09-Mar-2021.csv')
ansatz_state_f_exc = DataUtils.ansatz_from_data_frame(df_f_exc, molecule)
ansatz_f_exc = ansatz_state_f_exc.ansatz_elements
var_pars_f_exc = ansatz_f_exc.parameters

# <<<<<<<<<< INITIALISE DATAFRAME TO COLLECT RESULTS >>>>>>>>>>>>.
results_df = pd.DataFrame(columns=['prob_2', 'prob_meas', 'exp_value_q_exc', 'exp_value_f_exc'])
filename = '../../../results/zhenghao_testing/{}_noise_for_chem_acc_{}.csv' \
    .format(molecule.name, time_stamp)

# <<<<<<<<<<<< EXPECTATION VALUE FUNCTION CONFIGURATIONS>>>>>>>>>>>>>>>>>
n_shots = 1e5
method = 'automatic'

# <<<<<<<<<<<< EVALUATE HAM EXPECTATION VALUE >>>>>>>>>>>>>>>>>
expectation_qasm_ham = partial(TestUtils.expectation_qasm_ham,
                               hamiltonian=ham, n_qubits=n_qubits,
                               n_electrons=n_electrons, init_state_qasm=None,
                               n_shots=n_shots, coupling_map=None, method=method)

for idx in range(len(noise_model_list)):
    noise_model = noise_model_list[idx]
    prob_2 = prob_2_list[idx]
    prob_meas = prob_meas_list[idx]

    logging.info('')

    exp_value_q = expectation_qasm_ham(noise_model=noise_model,
                                       var_parameters=var_pars_q_exc,
                                       ansatz=ansatz_q_exc)
    exp_value_f = expectation_qasm_ham(noise_model=noise_model,
                                       var_parameters=var_pars_f_exc,
                                       ansatz=ansatz_f_exc)
    results_df.loc[idx] = {'prob_2': prob_2, 'prob_meas': prob_meas,
                           'exp_value_q_exc': exp_value_q,
                           'exp_value_f_exc': exp_value_f}
    results_df.to_csv(filename)

    message = 'Noise model is {}'.format(noise_model)
    logging.info(message)
    message = 'prob_2 = {}, prob_meas = {}'.format(prob_2, prob_meas)
    logging.info(message)
    message = 'exp_value_q_exc={}, exp_value_f_exc={}'.format(exp_value_q, exp_value_f)
