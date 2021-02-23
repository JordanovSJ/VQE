from src.q_systems import LiH, H2
from scripts.zhenghao.noisy_backends import QasmBackend
from src.ansatz_element_sets import UCCSD
from src.backends import QiskitSimBackend
from src.utils import *

from qiskit import QuantumCircuit, Aer, execute, IBMQ

import time
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
molecule = LiH(r)
n_qubits = molecule.n_qubits
n_electrons = molecule.n_electrons
hamiltonian = molecule.jw_qubit_ham
message = 'molecule={}, r={}, n_qubits={}, n_electrons={}'.format(molecule.name, r,
                                                                  n_qubits, n_electrons)
logging.info(message)

# <<<<<<<<<< ANSATZ >>>>>>>>>>>>.
uccsd = UCCSD(n_qubits, n_electrons)
ansatz = uccsd.get_excitations()
var_pars = np.zeros(len(ansatz))
while 0 in var_pars:
    var_pars = np.random.rand(len(ansatz))
qasm_ansatz = QasmBackend.qasm_from_ansatz(ansatz, var_pars)
init_state_qasm = QasmUtils.hf_state(molecule.n_electrons)
qasm_psi = init_state_qasm + qasm_ansatz

message = 'UCCSD ansatz with random var_pars'
logging.info(message)

# <<<<<<<<<< HAMILTONIAN TERM >>>>>>>>>>>>.
ham_terms = hamiltonian.terms
ham_keys_list = list(ham_terms.keys())
index = 10
operator_test = ham_keys_list[index]

message = 'Take the {}th term from the hamiltonian to test'.format(index)
logging.info(message)

# <<<<<<<<<< GENERATE QASM STRING >>>>>>>>>>>>.
t_q0 = time.time()
qasm_pauli = QasmBackend.pauli_convert(QasmBackend.qasm_from_op_key(operator_test))
header = QasmUtils.qasm_header(n_qubits)
qasm_to_eval = header + qasm_psi + qasm_pauli
t_q1 = time.time()

message = 'Time to generate qasm string from operator is {}'.format(t_q1 - t_q0)
logging.info(message)

# <<<<<<<<<< GENERATE CIRCUIT FROM QASM STRING >>>>>>>>>>>>.
t_c0 = time.time()
circ = QuantumCircuit.from_qasm_str(qasm_str=qasm_to_eval)
t_c1 = time.time()

message = 'Time to generate circuit from qasm string is {}'.format(t_c1 - t_c0)
logging.info(message)

# <<<<<<<<<< LOAD BACKEND FROM IBMQ >>>>>>>>>>>>.
t_b0 = time.time()
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
simulator = provider.get_backend('ibmq_qasm_simulator')
t_b1 = time.time()

message = 'Time to load backend from IBMQ is {}'.format(t_b1 - t_b0)
logging.info(message)

# <<<<<<<<<< LOAD BACKEND FROM AER >>>>>>>>>>>>.
t_aer0 = time.time()
simulator_aer = Aer.get_backend('qasm_simulator')
t_aer1 = time.time()
message = 'Time to load backend from Aer is {}'.format(t_aer1-t_aer0)
logging.info(message)

# <<<<<<<<<< TEST EVALUATION >>>>>>>>>>>>.
n_shots_list = [10, 100, 1000]

t_loop0 = time.time()
for n_shots in n_shots_list:
    message = 'n_shots={}'.format(n_shots)
    logging.info('')
    logging.info(message)

    t0 = time.time()
    exp_test = QasmBackend.eval_expectation_value(qasm_psi=qasm_psi, op_U=operator_test,
                                                  n_qubits=molecule.n_qubits, n_shots=n_shots)
    t1 = time.time()
    message = 'result={}, time={} for eval_exp_value with IBMQ backend'.format(exp_test, t1 - t0)
    logging.info(message)

    t2 = time.time()
    counts = QasmBackend.noisy_sim(qasm_str=qasm_to_eval, n_shots=n_shots)
    t3 = time.time()
    message = 'result = {}, time={} for noiseless_sim with IBMQ backend'.format(counts, t3 - t2)
    logging.info(message)

    t4 = time.time()
    job = execute(circ, simulator, shots=n_shots)
    t5 = time.time()
    job_id = job.job_id()
    message = 'job_id={}, time={} for execute with IBMQ backend'.format(job_id, t5-t4)
    logging.info(message)

    t6 = time.time()
    counts = QasmBackend.noisy_sim(qasm_str=qasm_to_eval, n_shots=n_shots)
    t7 = time.time()
    message = 'result = {}, time = {} for noiseless simulation with Aer backend'.format(counts, t7-t6)
    logging.info(message)

    t8 = time.time()
    job = execute(circ, simulator_aer, shots=n_shots)
    t9 = time.time()
    result = job.result()
    message = 'result status={}, time={} for execute with Aer backend'.format(result.status, t9-t8)
    logging.info(message)

t_loop1 = time.time()
message = 'For loop total time = {}'.format(t_loop1-t_loop0)
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
