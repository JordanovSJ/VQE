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
import numpy
import pandas as pd
import datetime
import qiskit


# <<<<<<<<<<<< MOLECULE >>>>>>>>>>>>>>>>>
r = 1
frozen_els = None #{'occupied': [0, 1], 'unoccupied': [6, 7]}
q_system = H4(r=r) #(r=r, frozen_els=frozen_els)

# <<<<<<<<<<<< LOGGING >>>>>>>>>>>>>>>>>
# logging
LogUtils.log_config()
message = 'H4 molecule, running single VQE optimisation for q_exc and f_exc based ansatz readily constructed by adapat vqe'
time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

# <<<<<<<<<<<< TUNABLE PARAMETERS >>>>>>>>>>>>>>>>>
ansatz_element_type = 'q_exc'
num_ansatz_element = 5  # Take only the first x ansatz elements

df_input = pd.read_csv('../../../results/iter_vqe_results/H4_adapt_vqe_q_exc_r=1_08-Mar-2021.csv')
ansatz_state = DataUtils.ansatz_from_data_frame(df_input, q_system)
ansatz = ansatz_state.ansatz_elements[0:num_ansatz_element]
# var_pars = [1e-4]*len(ansatz)  # ansatz_state.parameters
rest_pars = 0.01

prob_2 = 1e-6
time_cx = 0  # Gate time for cx gate

backend = QasmBackend
n_shots = 1e6
method = 'automatic'

optimizer = 'COBYLA'
# gtol = 10e-4
# adaptive_bool = True
optimizer_options = None # {'adaptive': adaptive_bool}  # {'gtol': gtol}

message = '{} type, prob_2={}, time_cx={}, backend={}, n_shots={}, method ={}, optimizer={}'\
    .format(ansatz_element_type, prob_2, time_cx, backend, n_shots, method, optimizer)
logging.info(message)

# message = 'Adaptive is {}'.format(adaptive_bool)
# logging.info(message)

# <<<<<<<<<<<< READING CSV FILES >>>>>>>>>>>>>>>>>

message = 'Length of {} based ansatz is {}'.format(ansatz_element_type, len(ansatz))
logging.info(message)

assert prob_2==1e-6  # for now no data for 1e-4
if prob_2==1e-6:
    # ref pars from '../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=1e-06_tcx=0_shots=1000000.0_18-Mar-2021 (19:12:16.969257).csv')
    ref_pars = [
        -0.20577359, -0.10015582, -0.08567858, -0.04775389, -0.05337564,
        -0.04962204, 0.06654893,  0.05192668,  0.03866926,  0.03087172,
        0.01216689]

    var_pars = ref_pars + [rest_pars]*(len(ansatz)-len(ref_pars))


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
# backend = QasmBackend
global_cache = None

message = 'Backend is {}, n_shots={}, method={}'.format(backend, n_shots, method)
logging.info(message)

# <<<<<<<<<<<< INITIALIZE DATA FRAME >>>>>>>>>>>>>>>>>
results_df = pd.DataFrame(columns=['iteration', 'energy', 'energy change', 'iteration duration', 'params'])
filename = '../../../results/zhenghao_testing/{}_vqe_{}_{}_p={}_tcx={}_{}_elements_shots={}_{}.csv' \
    .format(q_system.name, ansatz_element_type, optimizer, prob_2, time_cx, num_ansatz_element, n_shots, time_stamp)

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


