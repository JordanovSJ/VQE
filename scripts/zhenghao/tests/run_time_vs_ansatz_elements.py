import pandas as pd
import datetime
import time

from src.iter_vqe_utils import DataUtils
from src.molecules.molecules import H4
from scripts.zhenghao.noisy_backends import QasmBackend
from src.utils import *
from scripts.zhenghao.test_utils import NoiseUtils

r = 1
molecule = H4(r=r)
n_shots = 1e6

# <<<<<<<<<<<< LOGGING >>>>>>>>>>>>>>>>>
# logging
LogUtils.log_config()
message = 'H4 molecule, ' \
          'how does the run time for ham_expectation_value varies ' \
          'with number of ansatz elements? Run time is calculated on server.git '
logging.info(message)
time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

# <<<<<<<<<<<< ANSATZ >>>>>>>>>>>>>>>>>
df_input = pd.read_csv('../../../results/iter_vqe_results/H4_adapt_vqe_q_exc_r=1_08-Mar-2021.csv')
ansatz_state = DataUtils.ansatz_from_data_frame(df_input, molecule)
ansatz = ansatz_state.ansatz_elements
var_pars = [0.1] * len(ansatz)

cnot_num_list = df_input['cnot_count']

# <<<<<<<<<<<< Noise Model >>>>>>>>>>>>>>>>>
# Noise model
prob_2 = 1e-5
time_cx = 0  # Gate time for cx gate
prob_1 = 0  # Single qubit gate depolarizing error prob
prob_meas = prob_2
time_single_gate = 0  # Gate time for single qubit gate
time_meas = 0
t1 = 50e3  # T1 in nanoseconds
t2 = 50e3  # T2 in nanoseconds
noise_model = NoiseUtils.unified_noise(prob_1=prob_1, prob_2=prob_2, prob_meas=prob_meas,
                                       time_single_gate=time_single_gate, time_cx=time_cx,
                                       time_measure=time_meas, t1=t1, t2=t2)
coupling_map = None

message = 'Noise model generated for prob_1 = {}, prob_2={}, prob_meas={} ' \
          'time_single_gate={}, time_cx={}, time_meas={}, t1={}, t2={}. No coupling map.' \
    .format(prob_1, prob_2, prob_meas, time_single_gate, time_cx, time_meas, t1, t2)
logging.info(message)

# <<<<<<<<<<<< INITIALIZE DATA FRAME >>>>>>>>>>>>>>>>>
results_df = pd.DataFrame(columns=['element_num', 'noisy_energy',
                                   'noisy_time', 'noiseless_energy', 'noiseless_time'])
filename = '../../../results/zhenghao_testing/{}_run_time_vs_ansatz_elements_shots={}.csv' \
    .format(molecule.name, n_shots)

# <<<<<<<<<<<< EVALUATE EXPECTATION VALUE >>>>>>>>>>>>>>>>>
element_num_list = [1, 3, 5, 10]

idx = 0
for element_num in element_num_list:
    message = '{} ansatz elements'.format(element_num)
    logging.info('')
    logging.info(message)

    time_1 = time.time()
    exp_value_noisy = QasmBackend.ham_expectation_value(var_parameters=var_pars[0:element_num],
                                                        ansatz=ansatz[0:element_num],
                                                        q_system=molecule, n_shots=n_shots,
                                                        noise_model=noise_model, coupling_map=coupling_map)
    time_2 = time.time()
    time_noisy = time_2 - time_1
    message = 'Noisy, exp_value={}, time={}'.format(exp_value_noisy, time_noisy)
    logging.info(message)

    time_3 = time.time()
    exp_value_noiseless = QasmBackend.ham_expectation_value(var_parameters=var_pars[0:element_num],
                                                            ansatz=ansatz[0:element_num],
                                                            q_system=molecule, n_shots=n_shots,
                                                            noise_model=None, coupling_map=None)
    time_4 = time.time()
    time_noiseless = time_4 - time_3
    message = 'Noiseless, exp_value={}, time={}'.format(exp_value_noiseless,
                                                        time_noiseless)
    logging.info(message)

    results_df.loc[idx] = {'element_num': element_num, 'noisy_energy': exp_value_noisy,
                           'noisy_time': time_noisy,
                           'noiseless_energy': exp_value_noiseless,
                           'noiseless_time': time_noiseless}
    results_df.to_csv(filename)

    idx += 1
