import pandas as pd

from src.molecules.molecules import LiH, H2, H4
from scripts.zhenghao.noisy_backends import QasmBackend
from src.ansatz_element_sets import UCCSD
from src.backends import QiskitSimBackend
from src.utils import *

from qiskit import IBMQ
from qiskit.providers.aer.noise import NoiseModel

import time
import numpy as np

# qasm_test = '\ny q[0];\nz q[1];\nz q[2];\ny q[3];\nx q[4];\nx q[5];\n'

# <<<<<<<<<< MOLECULE >>>>>>>>>>>>.
r = 1.546
molecule = H2(r)
n_qubits = molecule.n_qubits
n_electrons = molecule.n_electrons
hamiltonian = molecule.jw_qubit_ham

# <<<<<<<<<< ANSATZ >>>>>>>>>>>>.
uccsd = UCCSD(n_qubits, n_electrons)
ansatz = uccsd.get_excitations()
var_pars = np.zeros(len(ansatz))
while 0 in var_pars:
    var_pars = np.random.rand(len(ansatz))

# <<<<<<<<<< LOGGING >>>>>>>>>>>>.
LogUtils.log_config()
logging.info('{}, r={} ,UCCSD ansatz'.format(molecule.name, r))
logging.info('n_qubits = {}, n_electrons = {}'.format(n_qubits, n_electrons))
time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

# <<<<<<<<<< NOISE MODEL >>>>>>>>>>>>.
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
backend = provider.get_backend('ibmq_16_melbourne')
noise_model = NoiseModel.from_backend(backend)
coupling_map = backend.configuration().coupling_map

message = 'Device noise generated from {}'.format(backend.name())
logging.info('')
logging.info(message)

# <<<<<<<<<< QiskitSimBackend >>>>>>>>>>>>.
qiskit_sim_result = QiskitSimBackend.ham_expectation_value(var_pars, ansatz, molecule)
message = 'QiskitSimBackend result is {}'.format(qiskit_sim_result)
logging.info('')
logging.info(message)

# <<<<<<<<<< QasmBackend >>>>>>>>>>>>.
n_shots_list = [10, 50, 100, 200, 500, 1000]
# n_shots_list = [10, 100, 1000]

# <<<<<<<<<< INITIALISE DATAFRAME TO COLLECT RESULTS >>>>>>>>>>>>.
# results_df = pd.DataFrame(columns=['n_shots', 'QiskitSimBackend',
#                                    'p_statevector', 'time_p_statevector',
#                                    'n_statevector', ' time_n_statevector',
#                                    'p_density_matrix', 'time_p_density_matrix',
#                                    'n_density_matrix', 'time_n_density_matrix'])
# filename_head = '../../results/zhenghao_testing/{}_r={}_UCCSD_qasm_backend_comparison'.format(molecule.name, r)
# filename_tail = '{}'.format(time_stamp)

method_list = [
    'statevector', 'density_matrix'
]

idx = 0
for n_shots in n_shots_list:
    message = 'n_shots = {}'.format(n_shots)
    logging.info('')
    logging.info(message)

    # results_df.loc[idx] = {'n_shots': n_shots, 'QiskitSimBackend': qiskit_sim_result,
    #                        'p_statevector': 0, 'time_p_statevector': 0,
    #                        'n_statevector': 0, ' time_n_statevector': 0,
    #                        'p_density_matrix': 0, 'time_p_density_matrix': 0,
    #                        'n_density_matrix': 0, 'time_n_density_matrix': 0}

    for method in method_list:
        message = 'method = {}'.format(method)
        logging.info(message)
        time_0 = time.time()
        expectation_value_1 = QasmBackend.ham_expectation_value(var_pars, ansatz, molecule,
                                                                n_shots=n_shots, noise_model=noise_model,
                                                                method=method, built_in_Pauli=False)
        time_1 = time.time()
        time_total = time_1 - time_0
        message = 'QasmBackend result is {} for manual Pauli, time used = {}s' \
            .format(expectation_value_1, time_total)
        logging.info(message)

        time_0 = time.time()
        expectation_value_1 = QasmBackend.ham_expectation_value(var_pars, ansatz, molecule,
                                                                n_shots=n_shots, noise_model=noise_model,
                                                                method=method,
                                                                built_in_Pauli=True)
        time_1 = time.time()
        time_total = time_1 - time_0
        message = 'QasmBackend result is {} for built in Pauli, time used = {}s' \
            .format(expectation_value_1, time_total)
        logging.info(message)

    idx += 1

# data = {'n_shots': n_shots_list, 'expectation_value': exp_val_list, 'time': time_list,
#         'ref_result': expectation_value_0}
# df = pandas.DataFrame(data)
# df.to_csv('csv_folder/{}_different_nshots_{}.csv'.format(molecule.name, time_stamp))

# qasm_test = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\ncreg c[4];\nx q[0];\nx q[1];\nh q[0];\ncx q[1], q[0];\nh q[0];\nh q[0];\nrz(1.5707963267948966) q[0];\ncx q[2], q[0];\nrz(-1.5707963267948966) q[0];\nrz(1.5707963267948966) q[2];\nh q[0];\nry(2.4059249570334487) q[2];\ncx q[0], q[2];\nry(-2.4059249570334487) q[2];\ncx q[2], q[0];\nh q[0];\ncx q[1], q[0];\nh q[0];\nh q[1];\ncx q[2], q[1];\nh q[1];\nh q[1];\nrz(1.5707963267948966) q[1];\ncx q[3], q[1];\nrz(-1.5707963267948966) q[1];\nrz(1.5707963267948966) q[3];\nh q[1];\nry(2.4533734055084206) q[3];\ncx q[1], q[3];\nry(-2.4533734055084206) q[3];\ncx q[3], q[1];\nh q[1];\ncx q[2], q[1];\nh q[1];\ncx q[0], q[1];\nx q[1];\ncx q[2], q[3];\nx q[3];\ncx q[0], q[2];\nrz(1.5707963267948966) q[0];\nrx(-0.0715007393139433) q[0];\nh q[1];\ncx q[0], q[1];\nh q[1];\nrx(0.0715007393139433) q[0];\nh q[3];\ncx q[0], q[3];\nh q[3];\nrx(-0.0715007393139433) q[0];\nh q[1];\ncx q[0], q[1];\nh q[1];\nrx(0.0715007393139433) q[0];\nh q[2];\ncx q[0], q[2];\nh q[2];\nrx(-0.0715007393139433) q[0];\nh q[1];\ncx q[0], q[1];\nh q[1];\nrx(0.0715007393139433) q[0];\nh q[3];\ncx q[0], q[3];\nh q[3];\nrx(-0.0715007393139433) q[0];\nh q[1];\ncx q[0], q[1];\nh q[1];\nrx(0.0715007393139433) q[0];\nrz(-1.5707963267948966) q[0];\nh q[2];\nrz(1.5707963267948966) q[2];\nrz(-1.5707963267948966) q[0];\ncx q[0], q[2];\nrz(-1.5707963267948966) q[2];\nh q[2];\nx q[1];\ncx q[0], q[1];\nx q[3];\ncx q[2], q[3];\n\nmeasure q[0] -> c[0];'
