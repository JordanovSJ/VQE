from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np
from scripts.zhenghao.noisy_backends import *
from scripts.zhenghao.utils_li import *
from scripts.zhenghao.test_utils import *
from qiskit.tools.visualization import plot_histogram

# <<<<<<<<<<<< CREATE QASM_STRING >>>>>>>>>>>>>>>>>
# Prepares qasm_psi
n_q = 2
qr = QuantumRegister(n_q)
circ_psi = QuantumCircuit(qr)
circ_psi.h(qr[0])
circ_psi.cx(qr[0], qr[1])
print('circ_psi')
print(circ_psi)
qasm_psi = circ_psi.qasm()

# Prepares qasm_U
qr_U = QuantumRegister(n_q)
circ_U = QuantumCircuit(qr_U)
circ_U.x(qr_U[0])
circ_U.h(qr_U[0])
circ_U.y(qr_U[1])
circ_U.cx(qr_U[1], qr_U[0])
print('circ_U')
print(circ_U)
qasm_U = circ_U.qasm()

# <<<<<<<<<<<< TEST QASM_COMPOSE >>>>>>>>>>>>>>>>>
qasm_psi_U_psi = QasmStrUtils.qasm_append(QasmStrUtils.qasm_append(qasm_psi, qasm_U),
                                          QasmStrUtils.qasm_invert(qasm_psi))
qasm_psi_U_psi = QasmStrUtils.qasm_measure_all(qasm_psi_U_psi)
circ_psi_U_psi = QuantumCircuit.from_qasm_str(qasm_psi_U_psi)
print('circ_psi_U_psi')
print(circ_psi_U_psi)

# <<<<<<<<<<<< CONSTRUCT NOISE MODELS >>>>>>>>>>>>>>>>>
vigo_noise_model, vigo_coupling_map = TestUtils.vigo_noise_model()
custom_noise_model = TestUtils.custom_noise_model()
thermal_noise_model = TestUtils.thermal_noise_model(n_q)

# <<<<<<<<<<<< TEST EXPECTATION EVALUATION FUNCTION >>>>>>>>>>>>>>>>>
exp_value_noiseless = NoisyBackend.eval_exp_value(qasm_psi=qasm_psi, qasm_U=qasm_U)
exp_value_noiseless_swap = NoisyBackend.eval_exp_value_swap(qasm_psi=qasm_psi, qasm_U=qasm_U)
print('noiseless exp value = {}'.format(exp_value_noiseless))
print('noiseless exp value swap = {}'.format(exp_value_noiseless_swap))

exp_value_vigo = NoisyBackend.eval_exp_value(qasm_psi=qasm_psi, qasm_U=qasm_U, noisy=True,
                                          noise_model=vigo_noise_model,
                                          coupling_map=vigo_coupling_map)
exp_value_vigo_swap = NoisyBackend.eval_exp_value_swap(qasm_psi=qasm_psi, qasm_U=qasm_U, noisy=True,
                                          noise_model=vigo_noise_model,
                                          coupling_map=vigo_coupling_map)
print('vigo noise exp value = {}'.format(exp_value_vigo))
print('vigo noise exp value swap = {}'.format(exp_value_vigo_swap))

exp_value_custom = NoisyBackend.eval_exp_value(qasm_psi=qasm_psi, qasm_U=qasm_U, noisy=True,
                                          noise_model=custom_noise_model)
print('custom noise exp value = {}'.format(exp_value_custom))

exp_value_thermal = NoisyBackend.eval_exp_value(qasm_psi=qasm_psi, qasm_U=qasm_U, noisy=True,
                                          noise_model=thermal_noise_model)
print('thermal noise exp value = {}'. format(exp_value_thermal))

counts_noiseless = NoisyBackend.noiseless_sim(qasm_psi_U_psi)
plot_histogram(counts_noiseless)

counts_vigo = NoisyBackend.noisy_sim(qasm_str=qasm_psi_U_psi, noise_model=vigo_noise_model,
                                     coupling_map=vigo_coupling_map)
plot_histogram(counts_vigo)