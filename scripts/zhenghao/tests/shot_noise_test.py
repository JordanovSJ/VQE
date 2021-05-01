from src.molecules.molecules import H4
from scripts.zhenghao.test_utils import NoiseUtils, TestUtils
from src.iter_vqe_utils import DataUtils
from src.utils import *
from src.backends import QiskitSimBackend
from scripts.zhenghao.noisy_backends import QasmBackend

from openfermion import QubitOperator

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
logging.info('{}, test shot noise against shot number'.format(molecule.name))
logging.info('n_qubits = {}, n_electrons = {}'.format(n_qubits, n_electrons))
time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

# <<<<<<<<<<<< TEST HAMILTONIAN >>>>>>>>>>>>>>>>>
# Take 5 random terms out of H4 hamiltonian
# ham = QubitOperator('Y0 Z1 Y2 X4 Z5 X6', 0.012709132884803409)
# ham += QubitOperator('Y0 Z1 Z2 X3 X4 Z5 Z6 Y7', 0.02318657944505642)
# ham += QubitOperator('Z0 Z1', 0.14266723882946233)
# ham += QubitOperator('Y3 Z4 Z5 Y7', 0.028916993734486442)
# ham += QubitOperator('Y1 Y2 X3 X4', -0.026485313086317092)

# message = 'Hamiltonian is {}'.format(ham)
ham = molecule.jw_qubit_ham
message = 'Hamiltonian is molecule.jw_qubit_ham'
logging.info(message)

# <<<<<<<<<<<< Noise Model >>>>>>>>>>>>>>>>>

noise_model = None
message = 'No quantum noise'
logging.info(message)

# <<<<<<<<<<<< READING CSV FILES >>>>>>>>>>>>>>>>>
df_q_exc = pd.read_csv('../../../results/iter_vqe_results/'
                       'H4_adapt_vqe_q_exc_r=1_08-Mar-2021.csv')
ansatz_state_q_exc = DataUtils.ansatz_from_data_frame(df_q_exc, molecule)
ansatz_q_exc = ansatz_state_q_exc.ansatz_elements
var_pars_q_exc = ansatz_state_q_exc.parameters

df_f_exc = pd.read_csv('../../../results/iter_vqe_results/'
                       'H4_adapt_vqe_eff_f_exc_r=1_09-Mar-2021.csv')
ansatz_state_f_exc = DataUtils.ansatz_from_data_frame(df_f_exc, molecule)
ansatz_f_exc = ansatz_state_f_exc.ansatz_elements
var_pars_f_exc = ansatz_state_f_exc.parameters

message = 'Entire Ansatz for both q and f'
logging.info(message)

# <<<<<<<<<<<< EXACT RESULTS >>>>>>>>>>>>>>>>>
exact_q_exc = QiskitSimBackend.ham_expectation_value(var_parameters=var_pars_q_exc,
                                                     ansatz=ansatz_q_exc,
                                                     q_system=molecule)
exact_f_exc = QiskitSimBackend.ham_expectation_value(var_parameters=var_pars_f_exc,
                                                     ansatz=ansatz_f_exc,
                                                     q_system=molecule)
message = 'QiskitSimBackend values are, for q_exc={}, for f_exc={}'.format(exact_q_exc,
                                                                            exact_f_exc)
logging.info(message)
# <<<<<<<<<< INITIALISE DATAFRAME TO COLLECT RESULTS >>>>>>>>>>>>.
results_df = pd.DataFrame(columns=['n_shots',
                                   'exact_q_exc', 'exact_f_exc',
                                   'exp_value_q_exc', 'exp_value_f_exc',
                                   'time_q_exc', 'time_f_exc'])
filename = '../../../results/zhenghao_testing/shot_noise/{}_shot_noise_test_{}.csv' \
    .format(molecule.name, time_stamp)

# <<<<<<<<<<<< EXPECTATION VALUE FUNCTION CONFIGURATIONS>>>>>>>>>>>>>>>>>
n_shot_list = [100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
method = 'automatic'
message = 'n_shots={}, method={}'.format(n_shot_list, method)
logging.info(message)

# <<<<<<<<<<<< EVALUATE HAM EXPECTATION VALUE >>>>>>>>>>>>>>>>>
# ham_expectation_value = partial(TestUtils.expectation_qasm_ham,
#                                hamiltonian=ham, n_qubits=n_qubits,
#                                n_electrons=n_electrons, init_state_qasm=None,
#                                n_shots=n_shots, coupling_map=None, method=method)

ham_expectation_value = partial(QasmBackend.ham_expectation_value, q_system=molecule,
                                init_state_qasm = None, method=method,)

idx = 0
for n_shots in n_shot_list:

    message = 'n_shots={}'.format(n_shots)
    logging.info('')
    logging.info(message)

    time_1 = time.time()
    exp_value_q = ham_expectation_value(noise_model=noise_model,
                                       var_parameters=var_pars_q_exc,
                                       ansatz=ansatz_q_exc, n_shots=n_shots)
    time_2 = time.time()
    time_q = time_2-time_1
    message = 'exp_value_q_exc={}, time_q_exc={}'.format(exp_value_q, time_q)
    logging.info(message)

    time_2 = time.time()
    exp_value_f = ham_expectation_value(noise_model=noise_model,
                                       var_parameters=var_pars_f_exc,
                                       ansatz=ansatz_f_exc, n_shots=n_shots)
    time_3 = time.time()
    time_f = time_3-time_2
    message = 'exp_value_f_exc={}, time_f_exc={}' \
        .format(exp_value_f, time_f)
    logging.info(message)

    results_df.loc[idx] = {'n_shots': n_shots,
                           'exact_q_exc': exact_q_exc,
                           'exact_f_exc': exact_f_exc,
                           'exp_value_q_exc': exp_value_q,
                           'exp_value_f_exc': exp_value_f,
                           'time_q_exc': time_q,
                           'time_f_exc': time_f}
    results_df.to_csv(filename)

    idx = idx + 1