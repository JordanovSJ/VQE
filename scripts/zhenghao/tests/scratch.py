from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.providers.aer import QasmSimulator
from qiskit.tools.visualization import plot_histogram

n_shots = 1e5

n_q = 2
qr = QuantumRegister(n_q)
circ = QuantumCircuit(qr)
circ.h(qr[0])
circ.cx(qr[0], qr[1])
circ.measure_all()

backend = QasmSimulator()
job = execute(circ, backend, shots=n_shots)
result = job.result()
counts = result.get_counts(circ)

plot_histogram(counts)