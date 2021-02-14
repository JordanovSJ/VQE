import pandas

from src.q_systems import LiH, H2, H4
from scripts.zhenghao.noisy_backends import QasmBackend
from src.ansatz_element_sets import UCCSD
from src.backends import QiskitSimBackend
from src.utils import *

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

# <<<<<<<<<< QiskitSimBackend >>>>>>>>>>>>.
expectation_value_0 = QiskitSimBackend.ham_expectation_value(var_pars, ansatz, molecule)
message = 'QiskitSimBackend result is {}'.format(expectation_value_0)

logging.info(message)

# <<<<<<<<<< QasmBackend >>>>>>>>>>>>.
n_shots_list = [1, 5, 10, 50, 100, 200, 500]
n_it = len(n_shots_list)
exp_val_list = np.zeros(n_it)
time_list = np.zeros(n_it)

i=0
for n_shots in n_shots_list:
    time_0 = time.time()
    expectation_value_1 = QasmBackend.ham_expectation_value(var_pars, ansatz, molecule, n_shots=n_shots)
    time_1 = time.time()
    time_total = time_1 - time_0

    message = 'QasmBackend result is {} for n_shots={}'.format(expectation_value_1, n_shots)

    logging.info(message)

    message_time = 'time used = {}s'.format(time_total)

    logging.info(message_time)

    exp_val_list[i] = expectation_value_1
    time_list[i] = time_total


data = {'n_shots': n_shots_list, 'expectation_value': exp_val_list, 'time': time_list}
df = pandas.DataFrame(data)
df.to_csv('csv_folder/{}_different_nshots_{}.csv'.format(molecule.name, time_stamp))

