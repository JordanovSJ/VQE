from src.vqe_runner import VQERunner
from src.q_system import *
from src.ansatz_element_sets import *
from src.backends import QiskitSimBackend, MatrixCacheBackend
from src.utils import LogUtils
from src.iter_vqe_utils import DataUtils
from src.cache import *
from src.molecules.molecules import H4, LiH

from scripts.zhenghao.noisy_backends import QasmBackend
from scripts.zhenghao.test_utils import NoiseUtils

import logging
import time
import numpy
import pandas as pd
import datetime
import qiskit


# <<<<<<<<<<<< MOLECULE >>>>>>>>>>>>>>>>>
r= 1.25
frozen_els = None
q_system = LiH(r=r)

# <<<<<<<<<<<< LOGGING >>>>>>>>>>>>>>>>>
# logging
LogUtils.log_config()
message = '{} molecule, running single VQE optimisation for q_exc ansatz readily constructed'.format(q_system.name)
time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

# <<<<<<<<<<<< TUNABLE PARAMETERS >>>>>>>>>>>>>>>>>
ansatz_element_type = 'q_exc'
num_ansatz_element = 5 # Take only the first x ansatz elements

# df_input = pd.read_csv('../../../results/iter_vqe_results/H4_adapt_vqe_q_exc_r=1_08-Mar-2021.csv')
df_input = pd.read_csv('../../../results/iter_vqe_results/LiH_iqeb_q_exc_r=1.25_19-Nov-2020.csv')
ansatz_state = DataUtils.ansatz_from_data_frame(df_input, q_system)
ansatz = ansatz_state.ansatz_elements[0:num_ansatz_element]
var_pars = [1e-2]*len(ansatz)  # ansatz_state.parameters
# var_pars = [-0.20577359, -0.10015582, -0.08567858, -0.04775389, -0.05337564, -0.04962204, 0.06654893, 0.05192668,
#             0.03866926, 0.03087172, 0.01216689, 0.01,0.01,0.01,0.01]
# var_pars = [-0.1999472,  -0.07384335, -0.0902148,  -0.04832507, -0.05559297,
#             -0.06442965, 0.05356275, 0.05492386, 0.03965827,  0.03441598,  0.01388335,
#             0.01047669, 0.01256415,  0.00931359,  0.00909889, 0.01, 0.01, 0.01, 0.01]
assert len(var_pars) == len(ansatz)

message = 'init_pars = {}'.format(var_pars)
logging.info(message)

prob_2 = 1e-5
time_cx = 0  # Gate time for cx gate

backend = QiskitSimBackend
n_shots = 1e6
method = 'automatic'

optimizer = 'COBYLA'
# optimizer_options = {'gtol': 10e-4}
# adaptive_bool=True
optimizer_options = {'maxiter': 500}
message = '{} type, prob_2={}, time_cx={}, backend={}, n_shots={}, method ={}, optimizer={}'\
    .format(ansatz_element_type, prob_2, time_cx, backend, n_shots, method, optimizer)
logging.info(message)

# message = 'Adaptive = {}'.format(adaptive_bool)
# logging.info(message)

# <<<<<<<<<<<< READING CSV FILES >>>>>>>>>>>>>>>>>

message = 'Length of {} based ansatz is {}'.format(ansatz_element_type, len(ansatz))
logging.info(message)

# <<<<<<<<<<<< Noise Model >>>>>>>>>>>>>>>>>
# Noise model
prob_1 = 0  # Single qubit gate depolarizing error prob
prob_meas = prob_2
time_single_gate = 0  # Gate time for single qubit gate
time_meas = 0
t1 = 50e3  # T1 in nanoseconds
t2 = 50e3  # T2 in nanoseconds
noise_model = NoiseUtils.unified_noise(prob_1=prob_1, prob_2=prob_2, prob_meas=prob_meas,
                                       time_single_gate=time_single_gate, time_cx = time_cx,
                                       time_measure=time_meas, t1=t1, t2=t2)
coupling_map = None

message = 'Noise model generated for prob_1 = {}, prob_2={}, prob_meas={} ' \
          'time_single_gate={}, time_cx={}, time_meas={}, t1={}, t2={}. No coupling map.' \
    .format(prob_1, prob_2, prob_meas, time_single_gate, time_cx, time_meas, t1, t2)
logging.info(message)


# <<<<<<<<<<<< BACKEND >>>>>>>>>>>>>>>>>
global_cache = None

message = 'Backend is {}, n_shots={}, method={}'.format(backend, n_shots, method)
logging.info(message)

# <<<<<<<<<<<< INITIALIZE DATA FRAME >>>>>>>>>>>>>>>>>
results_df = pd.DataFrame(columns=['iteration', 'energy', 'energy change', 'iteration duration', 'params'])
filename = '../../../results/zhenghao_testing/{}_vqe_{}_qiskitsim_{}_{}_elements_shots={}_{}.csv' \
    .format(q_system.name, ansatz_element_type, optimizer, num_ansatz_element, n_shots, time_stamp)

# <<<<<<<<<<<< VQE RUNNER >>>>>>>>>>>>>>>>>

vqe_runner = VQERunner(q_system, backend=backend, print_var_parameters=True,
                       use_ansatz_gradient=False,
                       optimizer=optimizer, optimizer_options=optimizer_options)

t0 = time.time()
result = vqe_runner.vqe_run(ansatz=ansatz, init_guess_parameters=var_pars, cache=global_cache,
                            n_shots=n_shots, noise_model=noise_model, coupling_map=coupling_map,
                            method=method, results_df=results_df, filename=filename)
t = time.time()

logging.critical(result)
print(result)
print('Time for running: ', t - t0)

print('Pizza')


