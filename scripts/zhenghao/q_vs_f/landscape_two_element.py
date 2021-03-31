from src.vqe_runner import VQERunner
from src.q_system import *
from src.ansatz_element_sets import *
from src.backends import QiskitSimBackend, MatrixCacheBackend
from src.utils import LogUtils
from src.iter_vqe_utils import DataUtils
from src.cache import *
from src.molecules.molecules import H4

from scripts.zhenghao.noisy_backends import QasmBackend
from scripts.zhenghao.test_utils import NoiseUtils

import logging
import time
import numpy as np
import pandas as pd
import datetime
import qiskit


# <<<<<<<<<<<< MOLECULE >>>>>>>>>>>>>>>>>
r = 1
frozen_els = None #{'occupied': [0, 1], 'unoccupied': [6, 7]}
q_system = H4(r=r) #(r=r, frozen_els=frozen_els)

n_shots = 1e6

# <<<<<<<<<<<< LOGGING >>>>>>>>>>>>>>>>>
# logging
LogUtils.log_config()
message = 'H4 molecule, ' \
          'investigate how the landscape should change with noise for first two ansatz elements of H4(r=1), {} shots'.format(n_shots)

logging.info(message)
time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")
# <<<<<<<<<<<< ANSATZ >>>>>>>>>>>>>>>>>
df_input = pd.read_csv('../../../results/iter_vqe_results/H4_adapt_vqe_q_exc_r=1_08-Mar-2021.csv')
ansatz_state = DataUtils.ansatz_from_data_frame(df_input, q_system)
ansatz = ansatz_state.ansatz_elements[0:2]
var_pars = [0]*len(ansatz)  # ansatz_state.parameters

message = 'Ansatz is {}'.format(ansatz)
logging.info(message)

# <<<<<<<<<<<< Noise Model >>>>>>>>>>>>>>>>>
# Noise model
prob_2 = 1e-6
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

# <<<<<<<<<<<< TIME STEP >>>>>>>>>>>>>>>>>
xlim = 0.25
ylim = 0.1

pixel_num_x = 50
pixel_num_y = 20

message = 'xlim={}, ylim={}, pixel number = {} x {}'.format(xlim, ylim, pixel_num_x, pixel_num_y)

logging.info(message)
# <<<<<<<<<<<< INTIALIZE DATAFRAME>>>>>>>>>>>>>>>>>
results_df = pd.DataFrame(columns=['pars_1', 'pars_2', 'qiskitsim_energy', 'qiskitsim_time',
                                   'noisy_energy', 'noisy_time'])
file_name = '../../../results/zhenghao_testing/h4_landscape_qiskit_sim_backend_{}.csv'.format(time_stamp)

idx = 0
for pars_1 in np.linspace(-xlim, xlim, num=pixel_num_x):
    for pars_2 in np.linspace(-ylim, ylim, num=pixel_num_y):
        var_pars=[pars_1, pars_2]

        time_1 = time.time()
        noiseless_energy = QiskitSimBackend.ham_expectation_value(var_parameters=var_pars, ansatz=ansatz,
                                                        q_system=q_system)
        time_2 = time.time()
        noiseless_time = time_2-time_1

        time_3 = time.time()
        noisy_energy = QasmBackend.ham_expectation_value(var_parameters=var_pars, ansatz=ansatz,
                                                         q_system=q_system, n_shots=n_shots,
                                                         noise_model=noise_model, coupling_map=coupling_map)
        time_4 = time.time()
        noisy_time = time_4-time_3

        results_df.loc[idx] = {'pars_1': pars_1, 'pars_2':pars_2,
                               'qiskitsim_energy': noiseless_energy, 'qiskitsim_time': noiseless_time,
                               'noisy_energy': noisy_energy, 'noisy_time': noisy_time}
        results_df.to_csv(file_name)

        message = '[{}, {}], qiskitsim_energy={}, qiskitsim_time={}, noisy_energy={}, noisy_time={}'\
            .format(pars_1, pars_2, noiseless_energy, noiseless_time, noisy_energy, noisy_time)
        logging.info(message)

        idx += 1
