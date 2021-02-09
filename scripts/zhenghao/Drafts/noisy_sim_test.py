import numpy as np

# Import Qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.tools.visualization import plot_histogram, plot_state_city
from qiskit import IBMQ
from qiskit.providers.aer.noise import device, NoiseModel
import qiskit.providers.aer.noise.errors as errors
from qiskit.providers.aer.noise.errors import pauli_error, reset_error, thermal_relaxation_error
from qiskit.providers.aer import QasmSimulator
from scripts.zhenghao.noisy_sim_fun import *

# <<<<<<<<<<<< CREATE QASM_STRING >>>>>>>>>>>>>>>>>
# Create a circuit to generate a qasm_string
n_q = 2
n_c = 2
qr = QuantumRegister(n_q, 'q_0')
cr = ClassicalRegister(n_c, 'c_0')
circ = QuantumCircuit(qr, cr)
circ.h(qr[0])
circ.cx(qr[0],qr[1])
circ.x(qr[0])
circ.h(qr[0])
circ.y(qr[1])
circ.cx(qr[1], qr[0])
circ.measure(qr, cr)
qasm_string = circ.qasm()

# <<<<<<<<<<<< IMPORT NOISE MODEL >>>>>>>>>>>>>>>>>
# Import noise model from ibmq_vigo backend
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
backend = provider.get_backend('ibmq_vigo')
noise_model_vigo = device.basic_device_noise_model(backend.properties())

# TO ASK: WHAT IS A COUPLING MAP?
coupling_map=backend.configuration().coupling_map

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
bit_flip = pauli_error([('X', prob_flip), ('I', 1-prob_flip)])  # Bit flip
phase_flip = pauli_error([('Z', prob_flip), ('I', 1-prob_flip)])  # Phase flip
bitphase_flip = bit_flip.compose(phase_flip)

# Compose the errors
error_gate1 = bitphase_flip.compose(error_1)  # For single qubit gates
error_gate2 = bitphase_flip.tensor(bitphase_flip).compose(error_2)  # For two qubit gates

# Measurement error: for a prob of p_meas the measurement of 1 gives 0 and 0 gives 1,
# i.e. a bit flip on measurement gate
p_meas = 0.1
error_meas = pauli_error([('X', p_meas), ('I', 1-p_meas)])

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
time_u1 = 0   # virtual gate
time_u2 = 50  # (single X90 pulse)
time_u3 = 100 # (two X90 pulses)
time_cx = 300
time_measure = 1000 # 1 microsecond

# QuantumError objects
errors_measure = [thermal_relaxation_error(t1, t2, time_measure)
                  for t1, t2 in zip(T1s, T2s)]
errors_u1  = [thermal_relaxation_error(t1, t2, time_u1)
              for t1, t2 in zip(T1s, T2s)]
errors_u2  = [thermal_relaxation_error(t1, t2, time_u2)
              for t1, t2 in zip(T1s, T2s)]
errors_u3  = [thermal_relaxation_error(t1, t2, time_u3)
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
# Run on noiseless simulator
counts = noiseless_sim(qasm_str=qasm_string, n_shots=2048)

# Run on noisy simulator
counts_vigo = noisy_sim(qasm_string, noise_model=noise_model_vigo,
                                            coupling_map=coupling_map, n_shots=2048)
counts_custom = noisy_sim(qasm_string, noise_model=noise_model_custom, n_shots=2048)
counts_thermal = noisy_sim(qasm_string, noise_model=noise_thermal, n_shots=2048)

# <<<<<<<<<<<< PLOTTING >>>>>>>>>>>>>>>>>
# Plotting the counts
plot_histogram(counts, title='Counts without noise')
plot_histogram(counts_vigo, title='Counts under ibmq-vigo noise')
plot_histogram(counts_custom, title='Counts under self customed noise')
plot_histogram(counts_thermal, title='Counts under thermal relaxation noise')