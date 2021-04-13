from src.backends import QiskitSimBackend
from src.utils import LogUtils
from src.iter_vqe_utils import DataUtils
from src.cache import *
from src.molecules.molecules import H4
from openfermion import commutator
from src.iter_vqe_utils import GradientUtils
from src.ansatz_element_sets import SDExcitations
from scripts.zhenghao.noisy_backends import QasmBackend
from scripts.zhenghao.test_utils import NoiseUtils, TestUtils

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
time_stamp = datetime.datetime.now().strftime("%d-%b-%Y")

# # <<<<<<<<<<<< QasmBackend >>>>>>>>>>>>>>>>>
# backend = QasmBackend
# n_shots = 1e-6
# method = 'automatic'
#
# message = 'Backend is {}, n_shots={}, method={}'.format(backend.__name__, n_shots, method)
# logging.info(message)

# <<<<<<<<<<<< QiskitSimBackend >>>>>>>>>>>>>>>>>
backend = QiskitSimBackend
n_shots = 1e-6 # placeholder
method = 'automatic' #placeholder

message = 'Backend is {}'.format(backend.__name__)
logging.info(message)

# <<<<<<<<<<<< NOISE >>>>>>>>>>>>>>>>>
prob_2 = 1e-4
time_cx = 0

if backend is QasmBackend:

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
else:
    noise_model = None #placeholder
    coupling_map = None #placeholder

    message = 'No noise model. No coupling map.'
    logging.info(message)


# <<<<<<<<<<<< ANSATZ >>>>>>>>>>>>>>>>>
df_q_input = pd.read_csv('../../../results/iter_vqe_results/H4_adapt_vqe_q_exc_r=1_08-Mar-2021.csv')
df_f_input = pd.read_csv('../../../results/iter_vqe_results/H4_adapt_vqe_eff_f_exc_r=1_09-Mar-2021.csv')

ansatz_dict = {
    'q_exc': DataUtils.ansatz_from_data_frame(df_q_input, q_system).ansatz_elements,
    'eff_f_exc': DataUtils.ansatz_from_data_frame(df_f_input, q_system).ansatz_elements
}

# <<<<<<<<<<<< PARAM DATA >>>>>>>>>>>>>>>>>
if prob_2==1e-4:
    param_dict = {
        'q_exc': pd.read_csv('../../../results/zhenghao_testing/fake_adapt_results/H4_q_exc_p2=1e-4_lowest_energy.csv'),
        'eff_f_exc': pd.read_csv('../../../results/zhenghao_testing/fake_adapt_results/H4_f_exc_p2=1e-4_lowest_energy.csv')
    }
elif prob_2==1e-6:
    param_dict = {
        'q_exc': pd.read_csv('../../../results/zhenghao_testing/fake_adapt_results/H4_q_exc_p2=1e-6_lowest_energy.csv'),
        'eff_f_exc': pd.read_csv('../../../results/zhenghao_testing/fake_adapt_results/H4_f_exc_p2=1e-6_lowest_energy.csv')
    }
else:
    raise Exception('No data for prob_2 = {}'.format(prob_2))

# <<<<<<<<<<<< ITERATION >>>>>>>>>>>>>>>>>
ansatz_element_type_list = ['q_exc', 'eff_f_exc']
num_element_list = [0, 5, 9, 12]

n_largest_grads = 10
message = 'Take {} largest grads'.format(n_largest_grads)
logging.info(message)

for num_element in num_element_list:

    message = 'Existing {} elements, find gradient for {}th element'.format(num_element, num_element+1)
    logging.info('')
    logging.info(message)

    for ansatz_element_type in ansatz_element_type_list:

        message = 'ansatz element type'.format(ansatz_element_type)
        logging.info('')
        logging.info(message)

        # <<<<<<<<<<<< INPUT ANSATZ >>>>>>>>>>>>>>>>>
        ansatz_input = ansatz_dict[ansatz_element_type]
        ansatz = ansatz_input[:num_element]

        # <<<<<<<<<<<< PARAMETERS >>>>>>>>>>>>>>>>>
        if num_element == 0:
            var_pars = [0]
        else:
            param_df = param_dict[ansatz_element_type]
            param_idx = list(param_df['num_elem']).index(num_element)
            var_pars = TestUtils.param_from_string(param_df['param'][param_idx])

        message = 'var pars = {}'.format(var_pars)
        logging.info(message)

        # <<<<<<<<<<<< Ansatz element pool >>>>>>>>>>>>>>>>>
        # create the pool of ansatz elements
        ansatz_element_pool = SDExcitations(q_system.n_orbitals, q_system.n_electrons,
                                            ansatz_element_type=ansatz_element_type).get_excitations()
        message = 'Length of Ansatz element pool is {}'.format(len(ansatz_element_pool))
        logging.info(message)

        # <<<<<<<<<<<< Getting the n largest elements >>>>>>>>>>>>>>>>>

        # get the n elements with largest gradients
        elements_grads = GradientUtils.\
                get_largest_gradient_elements(ansatz_element_pool, q_system,
                                              backend=backend, n=n_largest_grads,
                                              ansatz_parameters=var_pars,
                                              ansatz=ansatz,
                                              global_cache=None, excited_state=0,
                                              n_shots=n_shots, noise_model=noise_model,
                                              coupling_map=coupling_map, method=method)

        elements = [e_g[0] for e_g in elements_grads]
        grads = [e_g[1] for e_g in elements_grads]

        elements_names = [el.element for el in elements]
        elements_qubits = [el.qubits for el in elements]

        message = 'Elements with largest grads {}. Grads {}'.format(elements_names, grads)
        logging.info(message)

        # <<<<<<<<<<<< SAVING TO CSV FILE >>>>>>>>>>>>>>>>>
        results_df = pd.DataFrame(columns=['rank', 'gradient', 'element', 'element_qubits'])

        if backend is QiskitSimBackend:
            file_name = '../../../results/zhenghao_testing/ansatz_element_gradient/' \
                    '{}_{}_QiskitSim_p2={}_{}th_elem_{}.csv'.\
            format(q_system.name, ansatz_element_type, prob_2, num_element+1, time_stamp)
        else:
            file_name = '../../../results/zhenghao_testing/ansatz_element_gradient/' \
                        '{}_{}_p2={}_{}th_element_{}.csv'.\
                format(q_system.name, ansatz_element_type, prob_2, num_element+1, time_stamp)

        rank = np.flip(np.arange(1, n_largest_grads+1, 1))

        results_df['rank'] = rank
        results_df['element'] = elements_names
        results_df['element_qubits'] = elements_qubits
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

