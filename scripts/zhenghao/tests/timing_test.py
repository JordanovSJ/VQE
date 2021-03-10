from src.molecules.molecules import LiH, H4
from scripts.zhenghao.noisy_backends import QasmBackend
from src.ansatz_element_sets import UCCSD
from src.backends import QiskitSimBackend
from src.utils import *
from src.iter_vqe_utils import DataUtils

from qiskit import QuantumCircuit, Aer, execute, IBMQ
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel

import time
import pandas as pd
import numpy as np

# <<<<<<<<<< LOGGING >>>>>>>>>>>>.
time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")
logging_filename = 'time_test_{}'.format(time_stamp)
stdout_handler = logging.StreamHandler(sys.stdout)
logging.basicConfig(filename='../../../../results/zhenghao_testing/{}.txt'.format(logging_filename),
                    level=logging.INFO,
                    format='%(levelname)s %(asctime)s %(message)s')
# make logger print to console (it will not if multithreaded)
logging.getLogger().addHandler(stdout_handler)
# disable logging from qiskit
logging.getLogger('qiskit').setLevel(logging.WARNING)

# <<<<<<<<<< MOLECULE >>>>>>>>>>>>.
r = 1.546
molecule = H4(r)
n_qubits = molecule.n_qubits
n_electrons = molecule.n_electrons
hamiltonian = molecule.jw_qubit_ham
message = 'molecule={}, r={}, n_qubits={}, n_electrons={}'.format(molecule.name, r,
                                                                  n_qubits, n_electrons)
logging.info(message)

# <<<<<<<<<< READ CSV >>>>>>>>>>>>.
data_frame = pd.read_csv('../../../../results/iter_vqe_results/'
                         '{}_iqeb_q_exc_r={}_22-Feb-2021.csv'.format(molecule.name, r))
ansatz_state = DataUtils.ansatz_from_data_frame(data_frame, molecule)

# <<<<<<<<<< ANSATZ >>>>>>>>>>>>.
# uccsd = UCCSD(n_qubits, n_electrons)
# ansatz = uccsd.get_excitations()
# var_pars = np.zeros(len(ansatz))
# while 0 in var_pars:
#     var_pars = np.random.rand(len(ansatz))

ansatz = ansatz_state.ansatz_elements
var_pars = ansatz_state.parameters

message = 'IQEB found q_exc ansatz with random var_pars. ' \
          'Total {} ansatz elements'.format(len(ansatz))
logging.info(message)

# <<<<<<<<<< CONVERT ANSATZ TO QASM STRING >>>>>>>>>>>>.
num_ansatz_element = 1
t1 = time.time()
qasm_ansatz = QasmBackend.qasm_from_ansatz(ansatz[0:num_ansatz_element],
                                           var_pars[0:num_ansatz_element])
init_state_qasm = QasmUtils.hf_state(molecule.n_electrons)
qasm_psi = init_state_qasm + qasm_ansatz
t2 = time.time()

message = 'Use first {} ansatz elements. Convert into qasm string, time used={}s' \
    .format(num_ansatz_element, t2 - t1)
logging.info(message)

# <<<<<<<<<< HAMILTONIAN TERM >>>>>>>>>>>>.
ham_terms = hamiltonian.terms
ham_keys_list = list(ham_terms.keys())
index = 10
operator_test = ham_keys_list[index]

message = 'Take the {}th term from the hamiltonian to test. Total {} terms'.format(index, len(ham_keys_list))
logging.info(message)

# <<<<<<<<<< GENERATE QASM STRING FROM HAM TERM >>>>>>>>>>>>.
t_q0 = time.time()
qasm_pauli = QasmBackend.pauli_convert(QasmBackend.qasm_from_op_key(operator_test))
t_q1 = time.time()

message = 'Time to generate qasm string from ham term is {}'.format(t_q1 - t_q0)
logging.info(message)

# <<<<<<<<<< TOTAL QASM STRING >>>>>>>>>>>>.
header = QasmUtils.qasm_header(n_qubits)
qasm_to_eval = header + qasm_psi + qasm_pauli

# <<<<<<<<<< GENERATE CIRCUIT FROM QASM STRING >>>>>>>>>>>>.
t_c0 = time.time()
circ = QuantumCircuit.from_qasm_str(qasm_str=qasm_to_eval)
t_c1 = time.time()

message = 'Time to generate circuit from total qasm string is {}'.format(t_c1 - t_c0)
logging.info(message)

# <<<<<<<<<< NOISE MODEL >>>>>>>>>>>>.
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
backend = provider.get_backend('ibmq_16_melbourne')
noise_model = NoiseModel.from_backend(backend)
coupling_map = backend.configuration().coupling_map
basis_gates = noise_model.basis_gates

message = 'Noise model generated from {}'.format(backend.name())
logging.info(message)

# <<<<<<<<<< LOAD BACKEND FROM IBMQ >>>>>>>>>>>>.
# t_b0 = time.time()
# IBMQ.load_account()
# provider = IBMQ.get_provider(hub='ibm-q')
# simulator = provider.get_backend('ibmq_qasm_simulator')
# t_b1 = time.time()
#
# message = 'Time to load backend from IBMQ is {}'.format(t_b1 - t_b0)
# logging.info(message)

# <<<<<<<<<< LOAD BACKEND FROM AER >>>>>>>>>>>>.
method = 'automatic'

t_aer0 = time.time()
backend = QasmSimulator()
backend_options = {"method": method}
t_aer1 = time.time()

message = 'Time to load backend from Aer is {}. Method={}'.format(t_aer1 - t_aer0, method)
logging.info(message)

# <<<<<<<<<< TEST EVALUATION >>>>>>>>>>>>.
n_shots_list = [100]

for n_shots in n_shots_list:
    message = 'n_shots={}'.format(n_shots)
    logging.info('')
    logging.info(message)

    t1 = time.time()
    exp_value = QasmBackend.ham_expectation_value(var_parameters=var_pars[0:num_ansatz_element],
                                                  ansatz=ansatz[0:num_ansatz_element],
                                                  q_system=molecule, n_shots=n_shots,
                                                  method=method)
    t2 = time.time()
    message = 'Noiseless run full ham_expectation_value, exp_value={},' \
              ' for first {} ansatz elements, time used={}' \
        .format(exp_value, num_ansatz_element, t2 - t1)
    logging.info(message)

    t1 = time.time()
    exp_value = QasmBackend.ham_expectation_value(var_parameters=var_pars[0:num_ansatz_element],
                                                  ansatz=ansatz[0:num_ansatz_element],
                                                  q_system=molecule, n_shots=n_shots,
                                                  method=method, noise_model=noise_model,
                                                  coupling_map=coupling_map)
    t2 = time.time()
    message = 'Noisy run full ham_expectation_value, exp_value={},' \
              ' for first {} ansatz elements, time used={}' \
        .format(exp_value, num_ansatz_element, t2 - t1)
    logging.info(message)

    t6 = time.time()
    counts_1 = QasmBackend.noisy_sim(qasm_str=qasm_to_eval, n_shots=n_shots, method=method)
    t7 = time.time()
    message = 'Noiseless run noisy_sim, result counts= {}, time = {}, ' \
              'for test term ham and first {} ansatz elements' \
        .format(counts_1, t7 - t6, num_ansatz_element)
    logging.info(message)

    t6 = time.time()
    counts_2 = QasmBackend.noisy_sim(qasm_str=qasm_to_eval, noise_model=noise_model,
                                     coupling_map=coupling_map,
                                     n_shots=n_shots, method=method)
    t7 = time.time()
    message = 'Noisy run noisy_sim, result counts= {}, time = {}, ' \
              'for test term ham and first {} ansatz elements' \
        .format(counts_2, t7 - t6, num_ansatz_element)
    logging.info(message)

    t8 = time.time()
    job = execute(circ, backend, backend_options=backend_options, shots=n_shots)
    t9 = time.time()
    result_3 = job.result()
    counts_3 = result_3.get_counts(circ)
    message = 'Noiseless run execute with Aer backend, result counts= {}, time={},' \
              ' for test ham term and first {} ansatz elements' \
        .format(counts_3, t9 - t8, num_ansatz_element)
    logging.info(message)

    t8 = time.time()
    job_4 = execute(circ, backend, noise_model=noise_model,
                    basis_gates=basis_gates,
                    backend_options=backend_options, shots=n_shots,
                    coupling_map=coupling_map)
    t9 = time.time()
    result_4 = job_4.result()
    counts_4 = result_4.get_counts(circ)
    message = 'Noisy run execute with Aer backend, result counts= {}, time={},' \
              ' for test ham term and first {} ansatz elements' \
        .format(counts_4, t9 - t8, num_ansatz_element)
    logging.info(message)


# qasm_test = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\ncreg c[4];\nx q[0];\nx q[1];\nh q[0];\ncx q[1], q[0];\nh q[0];\nh q[0];\nrz(1.5707963267948966) q[0];\ncx q[2], q[0];\nrz(-1.5707963267948966) q[0];\nrz(1.5707963267948966) q[2];\nh q[0];\nry(2.4059249570334487) q[2];\ncx q[0], q[2];\nry(-2.4059249570334487) q[2];\ncx q[2], q[0];\nh q[0];\ncx q[1], q[0];\nh q[0];\nh q[1];\ncx q[2], q[1];\nh q[1];\nh q[1];\nrz(1.5707963267948966) q[1];\ncx q[3], q[1];\nrz(-1.5707963267948966) q[1];\nrz(1.5707963267948966) q[3];\nh q[1];\nry(2.4533734055084206) q[3];\ncx q[1], q[3];\nry(-2.4533734055084206) q[3];\ncx q[3], q[1];\nh q[1];\ncx q[2], q[1];\nh q[1];\ncx q[0], q[1];\nx q[1];\ncx q[2], q[3];\nx q[3];\ncx q[0], q[2];\nrz(1.5707963267948966) q[0];\nrx(-0.0715007393139433) q[0];\nh q[1];\ncx q[0], q[1];\nh q[1];\nrx(0.0715007393139433) q[0];\nh q[3];\ncx q[0], q[3];\nh q[3];\nrx(-0.0715007393139433) q[0];\nh q[1];\ncx q[0], q[1];\nh q[1];\nrx(0.0715007393139433) q[0];\nh q[2];\ncx q[0], q[2];\nh q[2];\nrx(-0.0715007393139433) q[0];\nh q[1];\ncx q[0], q[1];\nh q[1];\nrx(0.0715007393139433) q[0];\nh q[3];\ncx q[0], q[3];\nh q[3];\nrx(-0.0715007393139433) q[0];\nh q[1];\ncx q[0], q[1];\nh q[1];\nrx(0.0715007393139433) q[0];\nrz(-1.5707963267948966) q[0];\nh q[2];\nrz(1.5707963267948966) q[2];\nrz(-1.5707963267948966) q[0];\ncx q[0], q[2];\nrz(-1.5707963267948966) q[2];\nh q[2];\nx q[1];\ncx q[0], q[1];\nx q[3];\ncx q[2], q[3];\n\nmeasure q[0] -> c[0];'

# circ = QuantumCircuit.from_qasm_str(qasm_str=qasm_test)
# # simulator = Aer.get_backend('qasm_simulator')
#
# IBMQ.load_account()
# provider = IBMQ.get_provider(hub = 'ibm-q')
# simulator = provider.get_backend('ibmq_qasm_simulator')
# print('backend = {}'.format(simulator.name()))
#
# n_shots_list = [100]
#
#
# for n_shots in n_shots_list:
#     t0 = time.time()
#
#     job = execute(circ, simulator, shots=n_shots)
#     t_job = time.time()

# result = job.result()
# t_result = time.time()
#
# counts = result.get_counts(circ)
# t_counts = time.time()
#
# if '0001' in counts:
#     circ_result = counts['0001']/n_shots
# else:
#     circ_result = 0
#
# print('result = {}, time_job = {}, time_result={},'
#       'time_counts={}, n_shots = {}'.format(circ_result, t_job-t0, t_result-t_job,
#                                             t_counts-t_result, n_shots))
