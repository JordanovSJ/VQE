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

ansatz_element_type = 'q_exc'

num_elem_list = [1, 3, 8, 15]

for num_ansatz_element in num_elem_list:

    # <<<<<<<<<<<< LOGGING >>>>>>>>>>>>>>>>>
    # logging
    LogUtils.log_config()
    message = '{} molecule, running single VQE optimisation for q_exc ansatz readily constructed'.format(q_system.name)
    time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

    # <<<<<<<<<<<< READING CSV FILES >>>>>>>>>>>>>>>>>
    df_input = pd.read_csv('../../../results/iter_vqe_results/LiH_iqeb_q_exc_r=1.25_19-Nov-2020.csv')
    ansatz_state = DataUtils.ansatz_from_data_frame(df_input, q_system)
    ansatz = ansatz_state.ansatz_elements[0:num_ansatz_element]
    var_pars = [1e-2] * len(ansatz)  # ansatz_state.parameters

    message = 'init_pars = {}'.format(var_pars)
    logging.info(message)
    message = 'Length of {} based ansatz is {}'.format(ansatz_element_type, len(ansatz))
    logging.info(message)

    # <<<<<<<<<<<< Noise Model >>>>>>>>>>>>>>>>>
    # Noise model
    noise_model = None
    coupling_map = None

    message = 'Noise model is None'
    logging.info(message)

    # <<<<<<<<<<<< BACKEND >>>>>>>>>>>>>>>>>
    backend = QiskitSimBackend
    n_shots = 1e6
    method = 'automatic'
    global_cache = None

    message = 'Backend is {}, n_shots={}, method={}'.format(backend, n_shots, method)
    logging.info(message)

    # <<<<<<<<<<<< OPTIMIZER >>>>>>>>>>>>>>>>>
    optimizer = 'COBYLA'
    # optimizer_options = {'gtol': 10e-4}
    # adaptive_bool=True
    optimizer_options = {'maxiter': 500}
    message = '{} type, qiskitsim backend optimizer={}' \
        .format(ansatz_element_type, optimizer)
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




