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
logging.info('{}, finding standard deviation of measurements by evaluating ham expectation value'
             'with entire Q and F Ansatz by shot experiments'.format(molecule.name))
logging.info('n_qubits = {}, n_electrons = {}'.format(n_qubits, n_electrons))
time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

# <<<<<<<<<< HAMILTONIAN >>>>>>>>>>>>.
# message = 'Hamiltonian is {}'.format(ham)
ham = molecule.jw_qubit_ham
message = 'Hamiltonian is molecule.jw_qubit_ham'
logging.info(message)

# <<<<<<<<<<<< INPUT ANSATZ >>>>>>>>>>>>>>>>>
df_q_exc = pd.read_csv('../../../results/iter_vqe_results/'
                       'H4_adapt_vqe_q_exc_r=1_08-Mar-2021.csv')
ansatz_state_q_exc = DataUtils.ansatz_from_data_frame(df_q_exc, molecule)
ansatz_q_exc = ansatz_state_q_exc.ansatz_elements
var_pars_q = ansatz_state_q_exc.parameters

message = 'var pars q is {}'.format(var_pars_q)
logging.info(message)

df_f_exc = pd.read_csv('../../../results/iter_vqe_results/'
                       'H4_adapt_vqe_eff_f_exc_r=1_09-Mar-2021.csv')
ansatz_state_f_exc = DataUtils.ansatz_from_data_frame(df_f_exc, molecule)
ansatz_f_exc = ansatz_state_f_exc.ansatz_elements
var_pars_f = ansatz_state_f_exc.parameters

message = 'var pars f is {}'.format(var_pars_f)
logging.info(message)

# <<<<<<<<<<<< EXPECTATION VALUE FUNCTION CONFIGURATIONS>>>>>>>>>>>>>>>>>
n_shots = 100
method = 'automatic'
message = 'n_shots={}, method={}'.format(n_shots, method)
logging.info(message)

# <<<<<<<<<<<< Repeat measurements to find std dev >>>>>>>>>>>>>>>>>
num_repeat = 4096

message = 'Repeat measurements for {} times'.format(num_repeat)
logging.info(message)


ham_expectation_value = partial(QasmBackend.ham_expectation_value, q_system=molecule,
                                n_shots=n_shots, method=method,)


# <<<<<<<<<< INITIALISE DATAFRAME TO COLLECT RESULTS >>>>>>>>>>>>.
results_df = pd.DataFrame(columns=['i', 'q_exact', 'q_energy', 'q_time',
                                   'f_exact', 'f_energy', 'f_time'])
filename = '../../../results/zhenghao_testing/std_dev/' \
           'shot_noise_only/{}_{}shots_whole_ansatz_noiseless_params_repeat{}_{}.csv' \
    .format(molecule.name, n_shots, num_repeat, time_stamp)

exact_q = QiskitSimBackend.ham_expectation_value(var_parameters=var_pars_q,
                                                 ansatz=ansatz_q_exc,
                                                 q_system=molecule)
message = 'q_exact = {}'.format(exact_q)
logging.info(message)

exact_f = QiskitSimBackend.ham_expectation_value(var_parameters=var_pars_f,
                                                 ansatz=ansatz_f_exc,
                                                 q_system=molecule)
message = 'f_exact = {}'.format(exact_f)
logging.info(message)

for idx in range(1, num_repeat+1):

    message = 'i={}'.format(idx)
    logging.info('')
    logging.info(message)

    time_1 = time.time()
    exp_value_q = ham_expectation_value(var_parameters=var_pars_q,
                                        ansatz=ansatz_q_exc)
    time_2 = time.time()
    time_q = time_2 - time_1
    message = 'exp_value_q_exc={}, time_q_exc={}'.format(exp_value_q, time_q)
    logging.info(message)

    time_3 = time.time()
    exp_value_f = ham_expectation_value(var_parameters=var_pars_f,
                                        ansatz=ansatz_f_exc)
    time_4 = time.time()
    time_f = time_4 - time_3
    message = 'exp_value_f_exc={}, time_f_exc={}' \
        .format(exp_value_f, time_f)
    logging.info(message)

    results_df.loc[idx-1] = {
        'i': idx, 'q_exact': exact_q,'q_energy': exp_value_q, 'q_time': time_q,
        'f_exact': exact_f,'f_energy': exp_value_f, 'f_time': time_f
    }
    results_df.to_csv(filename)
