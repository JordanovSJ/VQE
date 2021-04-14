import logging

from src.molecules.molecules import H4
from scripts.zhenghao.test_utils import NoiseUtils, TestUtils
from src.iter_vqe_utils import DataUtils
from src.utils import *
from src.backends import QiskitSimBackend
from scripts.zhenghao.noisy_backends import QasmBackend

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
logging.info('{}, finding standard deviation of measurements'.format(molecule.name))
logging.info('n_qubits = {}, n_electrons = {}'.format(n_qubits, n_electrons))
time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

# <<<<<<<<<< HAMILTONIAN >>>>>>>>>>>>.
# message = 'Hamiltonian is {}'.format(ham)
ham = molecule.jw_qubit_ham
message = 'Hamiltonian is molecule.jw_qubit_ham'
logging.info(message)

# <<<<<<<<<<<< Noise Model >>>>>>>>>>>>>>>>>
# Noise model
prob_2 = 1e-4
prob_meas = prob_2

prob_1 = 0  # Single qubit gate depolarizing error prob
time_single_gate = 0  # Gate time for single qubit gate
time_cx = 0  # Gate time for cx gate
time_meas = 0
t1 = 50e3  # T1 in nanoseconds
t2 = 50e3  # T2 in nanoseconds

message = 'prob_1 = {}, prob_2={}, prob_meas={} time_single_gate={}, time_cx={}, time_meas={}, t1={}, t2={}' \
    .format(prob_1, prob_2, prob_meas, time_single_gate, time_cx, time_meas, t1, t2)
logging.info(message)


unified_noise = partial(NoiseUtils.unified_noise, prob_1=prob_1,
                        time_single_gate=time_single_gate,
                        time_cx=time_cx, time_measure=time_meas,
                        t1=t1, t2=t2)

noise_model = unified_noise(prob_2=prob_2, prob_meas=prob_meas)


# <<<<<<<<<<<< INPUT ANSATZ >>>>>>>>>>>>>>>>>
df_q_exc = pd.read_csv('../../../results/iter_vqe_results/'
                       'H4_adapt_vqe_q_exc_r=1_08-Mar-2021.csv')
ansatz_state_q_exc = DataUtils.ansatz_from_data_frame(df_q_exc, molecule)
ansatz_q_exc = ansatz_state_q_exc.ansatz_elements

df_f_exc = pd.read_csv('../../../results/iter_vqe_results/'
                       'H4_adapt_vqe_eff_f_exc_r=1_09-Mar-2021.csv')
ansatz_state_f_exc = DataUtils.ansatz_from_data_frame(df_f_exc, molecule)
ansatz_f_exc = ansatz_state_f_exc.ansatz_elements

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


# <<<<<<<<<<<< EXPECTATION VALUE FUNCTION CONFIGURATIONS>>>>>>>>>>>>>>>>>
n_shots = 100
method = 'automatic'
message = 'n_shots={}, method={}'.format(n_shots, method)
logging.info(message)

# <<<<<<<<<< INITIALISE DATAFRAME TO COLLECT RESULTS >>>>>>>>>>>>.
results_df = pd.DataFrame(columns=['i', 'q_energy', 'q_time', 'f_energy', 'f_time'])
filename = '../../../results/zhenghao_testing/std_dev/{}_p2={}_shots={}_{}.csv' \
    .format(molecule.name, prob_2, n_shots, time_stamp)

# <<<<<<<<<<<< Repeat measurements to find std dev >>>>>>>>>>>>>>>>>
num_repeat = 100

message = 'Repeat measurements for {} times'.format(num_repeat)
logging.info(message)

num_element_list = [10]
message = 'Measure std dev for {} elements'.format(num_element_list)
logging.info(message)

ham_expectation_value = partial(QasmBackend.ham_expectation_value, q_system=molecule,
                                init_state_qasm = None, n_shots=n_shots, method=method,)

for num_elem in num_element_list:
    message = 'num_elem = {}'.format(num_elem)
    logging.info('')
    logging.info(message)

    # <<<<<<<<<<<< PARAMETERS >>>>>>>>>>>>>>>>>
    param_q_df = param_dict['q_exc']
    param_q_idx = list(param_q_df['num_elem']).index(num_elem)
    var_pars_q_exc = TestUtils.param_from_string(param_q_df['param'][param_q_idx])
    message = 'var pars for q exc is {}'.format(var_pars_q_exc)
    logging.info(message)

    param_f_df = param_dict['eff_f_exc']
    param_f_idx = list(param_f_df['num_elem']).index(num_elem)
    var_pars_f_exc = TestUtils.param_from_string(param_f_df['param'][param_f_idx])
    message = 'var pars for f exc is {}'.format(var_pars_f_exc)
    logging.info(message)


    for idx in range(1, num_repeat+1):

        message = 'i={}'.format(idx)
        logging.info('')
        logging.info(message)

        time_1 = time.time()
        exp_value_q = ham_expectation_value(noise_model=noise_model,
                                            var_parameters=var_pars_q_exc,
                                            ansatz=ansatz_q_exc[:num_elem])
        time_2 = time.time()
        time_q = time_2 - time_1
        message = 'exp_value_q_exc={}, time_q_exc={}'.format(exp_value_q, time_q)
        logging.info(message)

        time_3 = time.time()
        exp_value_f = ham_expectation_value(noise_model=noise_model,
                                            var_parameters=var_pars_f_exc,
                                            ansatz=ansatz_f_exc[:num_elem])
        time_4 = time.time()
        time_f = time_4 - time_3
        message = 'exp_value_f_exc={}, time_f_exc={}' \
            .format(exp_value_f, time_f)
        logging.info(message)

        results_df.loc[idx-1] = {
            'i': idx, 'q_energy': exp_value_q, 'q_time': time_q,
            'f_energy': exp_value_f, 'f_time': time_f
        }
        results_df.to_csv(filename)
