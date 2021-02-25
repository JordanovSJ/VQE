import sys

sys.path.extend(['/home/adi/VQE'])

import pandas as pd
import time
from functools import partial

from qiskit import IBMQ
from qiskit.providers.aer.noise import NoiseModel

from src.molecules.molecules import LiH, H4
from src.iter_vqe_utils import DataUtils, QasmUtils
from src.utils import *
from scripts.zhenghao.noisy_backends import QasmBackend
from src.backends import QiskitSimBackend

# <<<<<<<<<< MOLECULE >>>>>>>>>>>>.
r = 1.546
molecule = H4(r=r)

# <<<<<<<<<< HAMILTONIAN >>>>>>>>>>>>.
hamiltonian = molecule.jw_qubit_ham
ham_terms = hamiltonian.terms
ham_keys_list = list(ham_terms.keys())
n_qubits = molecule.n_qubits
n_electrons = molecule.n_electrons

# <<<<<<<<<< LOGGING >>>>>>>>>>>>.
LogUtils.log_config()
logging.info('{}, r={}, iqeb_vqe result compare with noise'.format(molecule.name, r))
logging.info('n_qubits = {}, n_electrons = {}'.format(n_qubits, n_electrons))
time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

# <<<<<<<<<< READ CSV >>>>>>>>>>>>.
data_frame = pd.read_csv('../../results/iter_vqe_results/{}_iqeb_q_exc_r={}_22-Feb-2021.csv'.format(molecule.name, r))
reference_results = data_frame['E']
dE_list = data_frame['dE']
iteration_number = data_frame['n']
ansatz_state = DataUtils.ansatz_from_data_frame(data_frame, molecule)

# <<<<<<<<<< ANSATZ >>>>>>>>>>>>.
ansatz = ansatz_state.ansatz_elements
var_parameters = ansatz_state.parameters

# <<<<<<<<<< NOISE MODEL >>>>>>>>>>>>.
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
backend = provider.get_backend('ibmq_16_melbourne')
noise_model = NoiseModel.from_backend(backend)
coupling_map = backend.configuration().coupling_map

message = 'Noise model generated from {}'.format(backend.name())
# message = 'No noise'
logging.info(message)

# <<<<<<<<<< DIFFERENT METHODS >>>>>>>>>>>>.
# methods = ['statevector', 'density_matrix',
#            'matrix_product_state', 'automatic']
methods = ['statevector', 'statevector_gpu']

# <<<<<<<<<< INITIALISE DATAFRAME TO COLLECT RESULTS >>>>>>>>>>>>.
results_df = pd.DataFrame(columns=['n', 'noiseless E', 'noisy E', 'time'])
filename_head = '../../results/zhenghao_testing/{}_r={}_noise_comparison_'.format(molecule.name, r)
filename_tail = '{}'.format(time_stamp)

n_shots = 10  # Number of shots
threshold = 0.04  # Stopping point for dE

iter_index = 0

while dE_list[iter_index] > threshold:
    ansatz_index = iter_index + 1
    noiseless_result = reference_results[iter_index]
    iter_n = iteration_number[iter_index]

    message = 'Running QasmBackend.ham_expectation_value for {} molecule, ' \
              'first {} ansatz elements, noiseless value={}, {} shots' \
        .format(molecule.name, ansatz_index, noiseless_result, n_shots)
    logging.info(message)

    ham_expectation_value = partial(QasmBackend.ham_expectation_value,
                                    var_parameters=var_parameters[0:ansatz_index],
                                    ansatz=ansatz[0:ansatz_index],
                                    q_system=molecule, n_shots=n_shots,
                                    noise_model=noise_model, coupling_map=coupling_map)

    for method_str in methods:
        t0 = time.time()
        exp_value = ham_expectation_value(method=method_str)
        t1 = time.time()

        message = 'For {} method, noisy expectation value = {}, time = {}s, ' \
            .format(method_str, exp_value, t1 - t0)
        logging.info(message)

        results_df.loc[iter_index] = {'n': iter_n, 'noiseless E': noiseless_result,
                                      'noisy E': exp_value, 'time': t1-t0}
        filename = 'method={}_'.format(method_str) + filename_tail
        results_df.to_csv(filename_head + filename)

    iter_index += 1
