
from src.backends import QiskitSimBackend
from src.utils import LogUtils
from src.iter_vqe_utils import DataUtils
from src.cache import *
from src.molecules.molecules import H4
from openfermion import commutator
from src.iter_vqe_utils import GradientUtils
from src.ansatz_element_sets import SDExcitations
from scripts.zhenghao.noisy_backends import QasmBackend
from scripts.zhenghao.test_utils import NoiseUtils

import logging
import time
import numpy as np
import pandas as pd
import datetime
import qiskit

# <<<<<<<<<<<< MOLECULE >>>>>>>>>>>>>>>>>
q_system = H4(r=1)

# <<<<<<<<<<<< LOGGING >>>>>>>>>>>>>>>>>
# logging
LogUtils.log_config()
message = '{} molecule, Find ansatz element gradient'.format(q_system.name)
time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

# <<<<<<<<<<<< INPUT ANSATZ >>>>>>>>>>>>>>>>>
ansatz_element_type = 'eff_f_exc'
message = 'ansatz element type'.format(ansatz_element_type)
logging.info(message)

if ansatz_element_type == 'eff_f_exc':
    df_input = pd.read_csv('../../../results/iter_vqe_results/H4_adapt_vqe_eff_f_exc_r=1_09-Mar-2021.csv')
elif ansatz_element_type == 'q_exc':
    df_input = pd.read_csv('../../../results/iter_vqe_results/H4_adapt_vqe_q_exc_r=1_08-Mar-2021.csv')
else:
    raise Exception('ansatz element type {} not supported'.format(ansatz_element_type))

ansatz_state = DataUtils.ansatz_from_data_frame(df_input, q_system)
ansatz_input = ansatz_state.ansatz_elements

num_test = 5
# existing ansatz and var pars
ansatz = ansatz_input[:num_test]
var_pars = [0.01]*len(ansatz)

message = 'Take first 5 ansatz elements found by adapt vqe as ansatz, test the gradient for the 6th element'
logging.info(message)

# <<<<<<<<<<<< Ansatz element pool >>>>>>>>>>>>>>>>>
# create the pool of ansatz elements
ansatz_element_pool = SDExcitations(q_system.n_orbitals, q_system.n_electrons,
                                        ansatz_element_type=ansatz_element_type).get_excitations()
message = 'Length of new pool', len(ansatz_element_pool)
logging.info(message)

# <<<<<<<<<<<< BACKEND >>>>>>>>>>>>>>>>>
backend = QiskitSimBackend

message = 'backend is {}'.format(backend.__name__)
logging.info(message)


# <<<<<<<<<<<< Getting the n largest elements >>>>>>>>>>>>>>>>>
n_largest_grads = 10
message = 'take {} largest grads'.format(n_largest_grads)
logging.info(message)
# get the n elements with largest gradients
if backend is QiskitSimBackend:
    elements_grads = GradientUtils.\
        get_largest_gradient_elements(ansatz_element_pool, q_system,
                                      backend=backend, n=n_largest_grads,
                                      ansatz_parameters=var_pars,
                                      ansatz=ansatz, global_cache=None)

    elements = [e_g[0] for e_g in elements_grads]
    grads = [e_g[1] for e_g in elements_grads]

    elements_names = [el.element for el in elements]

message = 'Elements with largest grads {}. Grads {}'.format(elements_names, grads)
logging.info(message)

# <<<<<<<<<<<< SAVING TO CSV FILE >>>>>>>>>>>>>>>>>
results_df = pd.DataFrame(columns=['rank', 'ansatz_element', 'gradient'])
if backend is QiskitSimBackend:
    file_name = '../../../results/zhenghao_testing/ansatz_element_gradient/' \
            '{}_{}_{}th_element_largest_{}_gradients_{}.csv'.\
    format(q_system.name, backend.__name__, num_test, n_largest_grads, time_stamp)

rank = np.flip(np.arange(1, n_largest_grads+1, 1))

results_df['rank'] = rank
results_df['ansatz_element'] = elements_names
results_df['gradient'] = grads

results_df.to_csv(file_name)

# # the element of which we evaluate the gradient
# test_element = ansatz_input[num_test]
#
# # evaluate gradient by expectation value of the commutator on qasmbackend
# t1 = time.time()
# noisy_gradient = QasmBackend.ansatz_element_gradient(ansatz_element=test_element, var_parameters=var_pars, ansatz=ansatz,
#                                                      q_system=q_system)
# t2 = time.time()
# noisy_time = t2 - t1
# print('noisy time = {}'.format(noisy_time))
# # evaluate gradient on qiskitsimbackend
# exact_gradient = QiskitSimBackend.ansatz_element_gradient(ansatz_element=test_element, var_parameters=var_pars, ansatz=ansatz,
#                                                           q_system=q_system)
#
# print(noisy_gradient)
# print(exact_gradient)

