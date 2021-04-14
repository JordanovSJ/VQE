from qiskit.providers.aer.noise import device, NoiseModel
import qiskit.providers.aer.noise.errors as errors
from qiskit.providers.aer.noise.errors import pauli_error, thermal_relaxation_error
from qiskit import IBMQ, QuantumCircuit, execute
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram

# <<<<<<<<<<<< IMPORT NOISE MODEL >>>>>>>>>>>>>>>>>
# Import noise model from ibmq_vigo backend
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
backend = provider.get_backend('ibmq_16_melbourne')
noise_model_device = NoiseModel.from_backend(backend)
print(noise_model_device)

# <<<<<<<<<<<< CUSTOM NOISE MODEL >>>>>>>>>>>>>>>>>
# Create empty noise model
noise_model = NoiseModel()

# Depolarizing error for single and two qubit gates
prob_1 = 1e-4
prob_2 = 1e-3
error_1_dc = errors.depolarizing_error(prob_1, num_qubits=1)
error_2_dc = errors.depolarizing_error(prob_2, num_qubits=2)

# Measurement error: for a prob of p_meas the measurement of 1 gives 0 and 0 gives 1,
# i.e. a bit flip on measurement gate
prob_meas = 1e-3
error_meas_spam = pauli_error([('X', prob_meas), ('I', 1 - prob_meas)])

# Instruction times (in nanoseconds)
time_single_gate = 100
time_cx = 300
time_measure = 1000 # 1 microsecond

# T1 and T2 values
t1 = 50e3
t2 = 50e3

error_meas_therm = thermal_relaxation_error(t1, t2, time_measure)
error_2_therm = thermal_relaxation_error(t1, t2, time_cx).expand(
             thermal_relaxation_error(t1, t2, time_cx))
error_1_therm = thermal_relaxation_error(t1, t2, time_single_gate)

error_1 = error_1_dc.compose(error_1_therm)
error_2 = error_2_dc.compose(error_2_therm)
error_meas = error_meas_spam.compose(error_meas_therm)

# Add the errors to all qubits
noise_model.add_all_qubit_quantum_error(error_1, ['sx', 'x', 'id'])
noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
noise_model.add_all_qubit_quantum_error(error_meas, ['measure'])

print(noise_model)

# <<<<<<<<<<<< TEST CIRCUIT >>>>>>>>>>>>>>>>>

# System Specification
n_qubits = 4
circ = QuantumCircuit(n_qubits, n_qubits)

# Test Circuit
circ.h(0)
for qubit in range(n_qubits - 1):
    circ.cx(qubit, qubit + 1)
circ.measure(range(4), range(4))

# <<<<<<<<<<<< IDEAL SIMULATION >>>>>>>>>>>>>>>>>
ideal_simulator = QasmSimulator()
job = execute(circ, ideal_simulator)
result_ideal = job.result()
plot_histogram(result_ideal.get_counts(0))

# <<<<<<<<<<<< DEVICE NOISE SIMULATION >>>>>>>>>>>>>>>>>
# Run the noisy simulation
noisy_simulator = QasmSimulator(noise_model=noise_model_device)
job = execute(circ, noisy_simulator)
result_device = job.result()
counts_device = result_device.get_counts(0)

# Plot noisy output
plot_histogram(counts_device)

# <<<<<<<<<<<< CUSTOM NOISE SIMULATION >>>>>>>>>>>>>>>>>
# Run the noisy simulation
thermal_simulator = QasmSimulator(noise_model=noise_model)
job = execute(circ, thermal_simulator)
result_custom = job.result()
counts_custom = result_custom.get_counts(0)

# Plot noisy output
plot_histogram(counts_custom)

