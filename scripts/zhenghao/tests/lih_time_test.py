
from src.utils import LogUtils
from src.iter_vqe_utils import DataUtils
from src.molecules.molecules import LiH

from scripts.zhenghao.noisy_backends import QasmBackend
from scripts.zhenghao.test_utils import NoiseUtils

import logging
import time
import pandas as pd
import datetime

# <<<<<<<<<<<< MOLECULE >>>>>>>>>>>>>>>>>
r = 1.25
frozen_els = None  # {'occupied': [0, 1], 'unoccupied': [6, 7]}
q_system = LiH(r=r)  # (r=r, frozen_els=frozen_els)

# <<<<<<<<<<<< LOGGING >>>>>>>>>>>>>>>>>
# logging
LogUtils.log_config()
message = 'LiH molecule, to test the time for running vqe'
time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

# <<<<<<<<<<<< INPUT ANSATZ >>>>>>>>>>>>>>>>>
ansatz_element_type = 'q_exc'
# num_ansatz_element = 1  # Take only the first x ansatz elements

df_input = pd.read_csv('../../../results/iter_vqe_results/LiH_iqeb_q_exc_r=1.25_19-Nov-2020.csv')
ansatz_state = DataUtils.ansatz_from_data_frame(df_input, q_system)
# ansatz = ansatz_state.ansatz_elements[0:num_ansatz_element]
# var_pars = [0.01]*len(ansatz)

# <<<<<<<<<<<< Noise Model >>>>>>>>>>>>>>>>>
# Noise model
prob_2 = 1e-6
prob_1 = 0  # Single qubit gate depolarizing error prob
prob_meas = prob_2
time_cx = 10
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


# <<<<<<<<<<<< BACKEND >>>>>>>>>>>>>>>>>
backend = QasmBackend
n_shots = 1e6
method = 'automatic'

message = 'Backend is {}, n_shots={}, method={}'.format(backend, n_shots, method)
logging.info(message)

# <<<<<<<<<<<< INITIALIZE DATA FRAME >>>>>>>>>>>>>>>>>
results_df = pd.DataFrame(columns=['num_elem', 'energy', 'time'])
filename = '../../../results/zhenghao_testing/{}_{}_timetest_p={}_tcx={}_shots={}_{}.csv' \
    .format(q_system.name, ansatz_element_type, prob_2, time_cx, n_shots, time_stamp)


# <<<<<<<<<<<< LIST OF NUM OF ELEMENTS TO TEST >>>>>>>>>>>>>>>>>
num_elem_list = [1, 5, 8, 11, 15, 20]

idx = 0
for num_elem  in num_elem_list:
    ansatz = ansatz_state.ansatz_elements[0:num_elem]
    var_pars = [0.01]*num_elem

    message = '{} ansatz elements, var_pars = {}'.format(num_elem, var_pars)
    logging.info(message)

    t_1 = time.time()
    exp_val = QasmBackend.ham_expectation_value(var_parameters=var_pars, ansatz=ansatz,
                                                q_system=q_system, n_shots=n_shots,
                                                noise_model=noise_model, coupling_map=coupling_map,
                                                method=method)
    t_2 = time.time()

    message = 'Energy = {}, time = {}'.format(exp_val, t_2-t_1)
    logging.info(message)

    results_df.loc[idx] = {'num_elem': num_elem,
                           'energy': exp_val,
                           'time': t_2-t_1}
    results_df.to_csv(filename)

    idx +=1

