import pandas as pd

from src.molecules.molecules import LiH, H2, H4
from scripts.zhenghao.noisy_backends import QasmBackend
from src.ansatz_element_sets import UCCSD
from src.backends import QiskitSimBackend
from src.utils import *
from src.iter_vqe_utils import DataUtils

from qiskit import IBMQ
from qiskit.providers.aer.noise import NoiseModel

import time
import numpy as np

# qasm_test = '\ny q[0];\nz q[1];\nz q[2];\ny q[3];\nx q[4];\nx q[5];\n'

# <<<<<<<<<< MOLECULE >>>>>>>>>>>>.
r = 1.546
molecule = H4(r)
n_qubits = molecule.n_qubits
n_electrons = molecule.n_electrons
hamiltonian = molecule.jw_qubit_ham

# <<<<<<<<<< ANSATZ >>>>>>>>>>>>.
# uccsd = UCCSD(n_qubits, n_electrons)
# ansatz = uccsd.get_excitations()
# var_pars = np.zeros(len(ansatz))
# # while 0 in var_pars:
# #     var_pars = np.random.rand(len(ansatz))
# var_pars += 0.1
data_frame = pd.read_csv('../../results/iter_vqe_results/{}_iqeb_q_exc_r={}_22-Feb-2021.csv'.format(molecule.name, r))
ansatz_state = DataUtils.ansatz_from_data_frame(data_frame, molecule)
ansatz = ansatz_state.ansatz_elements
var_pars = ansatz_state.parameters
reference_results = data_frame['E']
cnot_count_ls = data_frame['cnot_count']
u1_count_ls = data_frame['u1_count']
cnot_depth_ls = data_frame['cnot_depth']
u1_depth_ls = data_frame['u1_depth']
 
# <<<<<<<<<< LOGGING >>>>>>>>>>>>.
LogUtils.log_config()

logging.info('{}, r={}, ansatz constructed from iqeb'.format(molecule.name, r))

logging.info('n_qubits = {}, n_electrons = {}'.format(n_qubits, n_electrons))
time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

# # <<<<<<<<<< NOISE MODEL >>>>>>>>>>>>.
# IBMQ.load_account()
# provider = IBMQ.get_provider(hub='ibm-q')
# backend = provider.get_backend('ibmq_16_melbourne')
# noise_model = NoiseModel.from_backend(backend)
# coupling_map = backend.configuration().coupling_map
# noise_model = None
# coupling_map = None

message = 'Device noise generated from {}'.format(backend.name())
logging.info('')
logging.info(message)

# <<<<<<<<<< QiskitSimBackend >>>>>>>>>>>>.
qiskit_sim_result = QiskitSimBackend.ham_expectation_value(var_pars, ansatz, molecule)
message = 'QiskitSimBackend result is {}, new calculation'.format(qiskit_sim_result)
logging.info('')
logging.info(message)

ref_size = len(reference_results)
ref_result = reference_results[ref_size - 1]
cnot_count = cnot_count_ls[ref_size-1]
u1_count = u1_count_ls[ref_size-1]
cnot_depth = cnot_depth_ls[ref_size-1]
u1_depth = u1_depth_ls[ref_size-1]
message = 'IQEB VQE reference result is {}'.format(ref_result)
logging.info('')
logging.info(message)
message = 'cnot_count={}, u1_count={}, cnot_depth={}, u1_depth={}'.format(cnot_count, u1_count, cnot_depth, u1_depth)
logging.info(message)

# <<<<<<<<<< QasmBackend >>>>>>>>>>>>.
n_shots_list = [1e2, 1e3, 1e4, 1e5, 1e6]
# n_shots_list = [10, 100, 1000]

# <<<<<<<<<< INITIALISE DATAFRAME TO COLLECT RESULTS >>>>>>>>>>>>.
results_df = pd.DataFrame(columns=['n_shots', 'ref E', 'noiseless E',
                                   'noiseless time',
                                   'noisy E', 'noisy time'])
filename_head = '../../results/zhenghao_testing/{}_r={}_iqeb_ansatz_compare_'.format(molecule.name, r)
filename_tail = '{}.csv'.format(time_stamp)

method_list = [
    'statevector'
]

idx = 0
for n_shots in n_shots_list:
    message = 'n_shots = {}'.format(n_shots)
    logging.info('')
    logging.info(message)

    for method in method_list:
        message = 'method = {}'.format(method)
        logging.info('')
        logging.info(message)

        time_0 = time.time()
        expectation_value_n = QasmBackend.ham_expectation_value(var_pars, ansatz, molecule,
                                                                n_shots=n_shots, noise_model=noise_model,
                                                                method=method,
                                                                built_in_Pauli=True)
        time_1 = time.time()
        time_n = time_1 - time_0
        message = 'QasmBackend result is {}, noisy, time used = {}s' \
            .format(expectation_value_n, time_n)
        logging.info(message)

        time_0 = time.time()
        expectation_value_p = QasmBackend.ham_expectation_value(var_pars, ansatz, molecule,
                                                                n_shots=n_shots,
                                                                method=method,
                                                                built_in_Pauli=True)
        time_1 = time.time()
        time_p = time_1 - time_0
        message = 'QasmBackend result is {}, noiseless, time used = {}s' \
            .format(expectation_value_p, time_p)
        logging.info(message)

        results_df.loc[idx] = {'n_shots': n_shots, 'ref E': ref_result,
                               'noiseless E': expectation_value_p,
                               'noiseless time': time_p,
                               'noisy E': expectation_value_n,
                               'noisy time': time_n}
        filename = 'method={}_'.format(method) + filename_tail
        results_df.to_csv(filename_head + filename)

    idx += 1
