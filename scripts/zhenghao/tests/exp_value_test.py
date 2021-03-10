from scripts.zhenghao.Drafts.noisy_sim_fun import *
from qiskit import IBMQ
from qiskit.providers.aer.noise import device, NoiseModel
import qiskit.providers.aer.noise.errors as errors
from qiskit.providers.aer.noise.errors import pauli_error, thermal_relaxation_error
from qiskit.quantum_info.operators import Operator

# <<<<<<<<<<<< CONSTRUCT CIRCUITS >>>>>>>>>>>>>>>>>
# Prepares qasm_psi
n_q = 2
qr = QuantumRegister(n_q, 'q')
circ_psi = QuantumCircuit(qr)
circ_psi.h(qr[0])
circ_psi.cx(qr[0], qr[1])
qasm_psi = circ_psi.qasm()

# Prepares qasm_U
qr_U = QuantumRegister(n_q, 'q_0')
circ_U = QuantumCircuit(qr_U)
circ_U.x(qr_U[0])
circ_U.h(qr_U[0])
circ_U.y(qr_U[1])
circ_U.cx(qr_U[1], qr_U[0])
qasm_U = circ_U.qasm()
op_h = Operator(circ_U)

# # To check if our functions return the right result, we manually construct U_psi* U U_psi|0>
# qr_test = QuantumRegister(n_q, 'q_test')
# cr_test = ClassicalRegister(n_q, 'c_test')
# circ_test = QuantumCircuit(qr_test, cr_test)
# circ_test.h(qr_test[0])
# circ_test.cx(qr_test[0],qr_test[1])
# circ_test.x(qr_test[0])
# circ_test.h(qr_test[0])
# circ_test.y(qr_test[1])
# circ_test.cx(qr_test[1], qr_test[0])
# circ_test.cx(qr_test[0],qr_test[1])
# circ_test.h(qr_test[0])
# circ_test.measure(qr_test, cr_test)
# counts_test = noiseless_sim_circ(circ_test)
# plot_histogram(counts_test, title="Counts manually constructed circuit")
#
# # And we construct the circuit by the circ_compose() method we wrote
# circ_test2 = circ_compose(circ_psi, circ_compose(circ_U, circ_psi.inverse()))
# qubits_test2 = circ_test2.qubits
# cr_test2 = ClassicalRegister(len(qubits_test2))
# circ_test2.add_register(cr_test2)
# circ_test2.measure(qubits_test2, cr_test2)
# counts_test2 = noiseless_sim_circ(circ_test2)
# plot_histogram(counts_test2, title="Counts circ_compose() method")


# <<<<<<<<<<<< IMPORT NOISE MODEL >>>>>>>>>>>>>>>>>
# Import noise model from ibmq_vigo backend
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
backend = provider.get_backend('ibmq_vigo')
noise_model_vigo = device.basic_device_noise_model(backend.properties())

# TO ASK: WHAT IS A COUPLING MAP?
coupling_map = backend.configuration().coupling_map

# <<<<<<<<<<<< SELF CUSTOM NOISE MODEL >>>>>>>>>>>>>>>>>
# Create empty noise model
noise_model_custom = NoiseModel()

# Depolorizing errors for single and two qubit gates
prob_1 = 0.001  # Error prob for single qubit gate
prob_2 = 0.01  # Error prob for two qubit gates
error_1 = errors.depolarizing_error(prob_1, num_qubits=1)  # For single qubit gates
error_2 = errors.depolarizing_error(prob_2, num_qubits=2)  # For two qubit gates

# Pauli error
prob_flip = 0.05
bit_flip = pauli_error([('X', prob_flip), ('I', 1 - prob_flip)])  # Bit flip
phase_flip = pauli_error([('Z', prob_flip), ('I', 1 - prob_flip)])  # Phase flip
bitphase_flip = bit_flip.compose(phase_flip)

# Compose the errors
error_gate1 = bitphase_flip.compose(error_1)  # For single qubit gates
error_gate2 = bitphase_flip.tensor(bitphase_flip).compose(error_2)  # For two qubit gates

# Measurement error: for a prob of p_meas the measurement of 1 gives 0 and 0 gives 1,
# i.e. a bit flip on measurement gate
p_meas = 0.1
error_meas = pauli_error([('X', p_meas), ('I', 1 - p_meas)])

# Add errors to the noise model
noise_model_custom.add_all_qubit_quantum_error(error_gate1, ['u1', 'u2', 'u3'])
noise_model_custom.add_all_qubit_quantum_error(error_gate2, ['cx'])
noise_model_custom.add_all_qubit_quantum_error(error_meas, ['measure'])

# <<<<<<<<<<<< Thermal relaxation errors >>>>>>>>>>>>>>>>>
# Create empty noise model
noise_thermal = NoiseModel()

# T1 and T2 values for qubits 0-3
T1s = np.random.normal(50e3, 10e3, n_q)  # Sampled from normal distribution mean 50 microsec
T2s = np.random.normal(70e3, 10e3, n_q)  # Sampled from normal distribution mean 50 microsec

# Truncate random T2s <= T1s
T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(n_q)])

# Instruction times (in nanoseconds)
time_u1 = 0  # virtual gate
time_u2 = 50  # (single X90 pulse)
time_u3 = 100  # (two X90 pulses)
time_cx = 300
time_measure = 1000  # 1 microsecond

# QuantumError objects
errors_measure = [thermal_relaxation_error(t1, t2, time_measure)
                  for t1, t2 in zip(T1s, T2s)]
errors_u1 = [thermal_relaxation_error(t1, t2, time_u1)
             for t1, t2 in zip(T1s, T2s)]
errors_u2 = [thermal_relaxation_error(t1, t2, time_u2)
             for t1, t2 in zip(T1s, T2s)]
errors_u3 = [thermal_relaxation_error(t1, t2, time_u3)
             for t1, t2 in zip(T1s, T2s)]
errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(
    thermal_relaxation_error(t1b, t2b, time_cx))
    for t1a, t2a in zip(T1s, T2s)]
    for t1b, t2b in zip(T1s, T2s)]

for j in range(n_q):
    noise_thermal.add_quantum_error(errors_measure[j], "measure", [j])
    noise_thermal.add_quantum_error(errors_u1[j], "u1", [j])
    noise_thermal.add_quantum_error(errors_u2[j], "u2", [j])
    noise_thermal.add_quantum_error(errors_u3[j], "u3", [j])
    for k in range(n_q):
        noise_thermal.add_quantum_error(errors_cx[j][k], "cx", [j, k])

# <<<<<<<<<<<< EXECUTION >>>>>>>>>>>>>>>>>
n_shots = 2048
# Run on noiseless simulator
exp_val_noiseless = eval_exp_value(qasm_psi=qasm_psi, qasm_U=qasm_U, n_shots=n_shots)
exp_val_2_noiseless = eval_exp_value_swap(qasm_psi=qasm_psi, qasm_U=qasm_U, n_shots=n_shots)
ham_val_noiseless = ham_exp_value(qasm_psi, op_h, n_shots)
ham_val_2_noiseless = ham_exp_value_swap(qasm_psi, op_h, n_shots)
# Run on noisy simulator
# Vigo noise model
exp_val_vigo = eval_exp_value(qasm_psi=qasm_psi, qasm_U=qasm_U, n_shots=n_shots, noisy=True,
                              noise_model=noise_model_vigo, coupling_map=coupling_map)
ham_val_vigo = ham_exp_value(qasm_psi=qasm_psi, op_h=op_h, n_shots=n_shots, noisy=True,
                             noise_model=noise_model_vigo, coupling_map=coupling_map)
exp_val_2_vigo = eval_exp_value_swap(qasm_psi=qasm_psi, qasm_U=qasm_U, n_shots=n_shots, noisy=True,
                                     noise_model=noise_model_vigo, coupling_map=coupling_map)
ham_val_2_vigo = ham_exp_value_swap(qasm_psi=qasm_psi, op_h=op_h, n_shots=n_shots, noisy=True,
                                    noise_model=noise_model_vigo, coupling_map=coupling_map)

# Custom noise model
exp_val_custom = eval_exp_value(qasm_psi=qasm_psi, qasm_U=qasm_U, n_shots=n_shots, noisy=True,
                                noise_model=noise_model_custom)
ham_val_custom = ham_exp_value(qasm_psi=qasm_psi, op_h=op_h, n_shots=n_shots, noisy=True,
                               noise_model=noise_model_custom)
exp_val_2_custom = eval_exp_value_swap(qasm_psi=qasm_psi, qasm_U=qasm_U, n_shots=n_shots, noisy=True,
                                       noise_model=noise_model_custom)
ham_val_2_custom = eval_exp_value_swap(qasm_psi=qasm_psi, qasm_U=qasm_U, n_shots=n_shots, noisy=True,
                                       noise_model=noise_model_custom)

# Thermal noise model
exp_val_thermal = eval_exp_value(qasm_psi=qasm_psi, qasm_U=qasm_U, n_shots=n_shots, noisy=True,
                                 noise_model=noise_thermal)
ham_val_thermal = ham_exp_value(qasm_psi=qasm_psi, op_h=op_h, n_shots=n_shots, noisy=True,
                                noise_model=noise_thermal)
exp_val_2_thermal = eval_exp_value_swap(qasm_psi=qasm_psi, qasm_U=qasm_U, n_shots=n_shots, noisy=True,
                                        noise_model=noise_thermal)
ham_val_2_thermal = ham_exp_value_swap(qasm_psi=qasm_psi, op_h=op_h, n_shots=n_shots, noisy=True,
                                       noise_model=noise_thermal)

print("noiseless: {value}".format(value=exp_val_noiseless))
print("ham noiseless: {value}".format(value=ham_val_noiseless))

print("vigo: {value}".format(value=exp_val_vigo))
print("ham vigo: {value}".format(value=ham_val_vigo))

print("custom: {value}".format(value=exp_val_custom))
print("ham custom: {value}".format(value=ham_val_custom))

print("thermal: {value}".format(value=exp_val_thermal))
print("ham thermal: {value}".format(value=ham_val_thermal))

print("noiseless swap: {value}".format(value=exp_val_2_noiseless))
print("ham noiseless swap: {value}".format(value=ham_val_2_noiseless))

print("vigo swap: {value}".format(value=exp_val_2_vigo))
print("ham vigo swap: {value}".format(value=ham_val_2_vigo))

print("custom swap: {value}".format(value=exp_val_2_custom))
print("ham custom swap: {value}".format(value=ham_val_2_custom))

print("thermal swap: {value}".format(value=exp_val_2_thermal))
print("ham thermal swap: {value}".format(value=ham_val_2_thermal))

