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

# <<<<<<<<<<<< READING CSV FILES >>>>>>>>>>>>>>>>>
df_q_exc = pd.read_csv('../../../results/iter_vqe_results/'
                       'H4_adapt_vqe_q_exc_r=1_08-Mar-2021.csv')
ansatz_state_q_exc = DataUtils.ansatz_from_data_frame(df_q_exc, q_system)
ansatz = ansatz_state_q_exc.ansatz_elements
var_pars = ansatz_state_q_exc.parameters

message = 'Length of q_exc based ansatz is {}'.format(len(ansatz))
logging.info(message)


# <<<<<<<<<<<< Noise Model >>>>>>>>>>>>>>>>>
# Noise model
prob_1 = 0  # Single qubit gate depolarizing error prob
prob_2 = 1e-5
prob_meas = prob_2
time_single_gate = 0  # Gate time for single qubit gate
time_cx = 0  # Gate time for cx gate
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
backend = QiskitSimBackend
global_cache = None

n_shots = 1e6
method = 'automatic'

message = 'Backend is {}, n_shots={}, method={}'.format(backend, n_shots, method)

# <<<<<<<<<<<< VQE RUNNER >>>>>>>>>>>>>>>>>
optimizer = 'BFGS'
optimizer_options = {'gtol': 10e-8}
vqe_runner = VQERunner(q_system, backend=backend, print_var_parameters=True,
                       use_ansatz_gradient=False,
                       optimizer=optimizer, optimizer_options=optimizer_options)

t0 = time.time()
result = vqe_runner.vqe_run(ansatz=ansatz, init_guess_parameters=var_pars, cache=global_cache,
                            n_shots=n_shots, noise_model=noise_model, coupling_map=coupling_map,
                            method=method)
t = time.time()

logging.critical(result)
print(result)
print('Time for running: ', t - t0)

print('Pizza')


