from qiskit import QuantumCircuit, QuantumRegister, Aer, transpile, assemble

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error

from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter

def get_noise(p):
    error_meas = pauli_error([('X', p), ('I', 1 - p)])

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure")  # measurement error is applied to measurements

    return noise_model

qasm_sim = Aer.get_backend('qasm_simulator')
noise_model = get_noise(0.1)

qr = QuantumRegister(2)
meas_calibs, state_labels = complete_meas_cal(qr=qr, circlabel='mcal')

t_qc = transpile(meas_calibs, qasm_sim)
qobj = assemble(t_qc, shots=10000)
cal_results = qasm_sim.run(qobj, noise_model=noise_model, shots=10000).result()

meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal')
print('cal matrix= {}'.format(meas_fitter.cal_matrix))


qc = QuantumCircuit(2,2)
qc.h(0)
qc.cx(0,1)
qc.measure(qc.qregs[0],qc.cregs[0])

t_qc = transpile(qc, qasm_sim)
qobj = assemble(t_qc, shots=10000)
results = qasm_sim.run(qobj, noise_model=noise_model, shots=10000).result()
print('Noisy result={}'.format(results.get_counts()))

# Get the filter object
meas_filter = meas_fitter.filter

# Results with mitigation
mitigated_results = meas_filter.apply(results)
mitigated_counts = mitigated_results.get_counts(0)
print('Mitigated result={}'.format(mitigated_counts))