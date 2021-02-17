import pandas as pd
import time

from qiskit import IBMQ
from qiskit.providers.aer.noise import device, NoiseModel

from src.q_systems import LiH
from src.iter_vqe_utils import DataUtils, QasmUtils
from src.utils import *
from scripts.zhenghao.noisy_backends import QasmBackend

# <<<<<<<<<< MOLECULE >>>>>>>>>>>>.
r = 3.25
molecule = LiH(r=r)

# <<<<<<<<<< HAMILTONIAN >>>>>>>>>>>>.
hamiltonian = molecule.jw_qubit_ham
ham_terms = hamiltonian.terms
ham_keys_list = list(ham_terms.keys())
n_qubits = molecule.n_qubits
n_electrons = molecule.n_electrons

test_index = 50
test_ham_term = ham_keys_list[test_index]

# <<<<<<<<<< LOGGING >>>>>>>>>>>>.
LogUtils.log_config()
logging.info('{}, r={}, iqeb_vqe result'.format(molecule.name, r))
logging.info('n_qubits = {}, n_electrons = {}'.format(n_qubits, n_electrons))
time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

# <<<<<<<<<< READ CSV >>>>>>>>>>>>.
data_frame = pd.read_csv('../../results/iter_vqe_results/LiH_iqeb_q_exc_r={}_19-Nov-2020.csv'.format(r))
reference_results = data_frame['E']
ansatz_state = DataUtils.ansatz_from_data_frame(data_frame, molecule)

# <<<<<<<<<< ANSATZ >>>>>>>>>>>>.
ansatz = ansatz_state.ansatz_elements
var_parameters = ansatz_state.parameters

# # <<<<<<<<<< QASM_PSI >>>>>>>>>>>>.
# init_state_qasm = QasmUtils.hf_state(molecule.n_electrons)
# ansatz_index = 1
# qasm_ansatz = QasmBackend.qasm_from_ansatz(ansatz[0:ansatz_index], var_parameters[0:ansatz_index])
# qasm_psi = init_state_qasm + qasm_ansatz

# <<<<<<<<<< NOISE MODEL >>>>>>>>>>>>.
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
backend = provider.get_backend('ibmq_16_melbourne')
noise_model = device.basic_device_noise_model(backend.properties())
coupling_map = backend.configuration().coupling_map

message = 'Noise model generated from {}'.format(backend.name())
logging.info(message)

index_list = [1, 10]
n_shots = 10

for ansatz_index in index_list:
    message = 'Running ham_expectation_value for {} molecule, first {} ansatz elements, {} shots' \
        .format(molecule.name, ansatz_index, n_shots)
    logging.info(message)

    t0 = time.time()
    exp_value = QasmBackend.ham_expectation_value(var_parameters=var_parameters[0:ansatz_index],
                                                  ansatz=ansatz[0:ansatz_index],
                                                  q_system=molecule, n_shots=n_shots,
                                                  noisy=True,
                                                  noise_model=noise_model, coupling_map=coupling_map)
    t1 = time.time()
    message = 'Expectation value = {}, time = {}s, Yordan result = {}' \
        .format(exp_value, t1 - t0, reference_results[ansatz_index-1])
    logging.info(message)



# t2 = time.time()
# test_exp_value = QasmBackend.eval_expectation_value(qasm_psi=qasm_psi, op_U=test_ham_term,
#                                                     n_qubits=n_qubits, n_shots=10)
# t3 = time.time()
# message = 'Expectation value = {}, molecule = {}, first {} ansatz elements,' \
#           ' {}th term in hamiltonian, time = {}'.format(test_exp_value, molecule.name, ansatz_index,
#                                                         test_index, t2-t1)
# print(message)
