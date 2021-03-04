from src.molecules.molecules import H2
from scripts.zhenghao.noisy_backends import QasmBackend
from src.utils import QasmUtils

from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import PauliExpectation, CircuitSampler, StateFn, CircuitStateFn
from qiskit import QuantumCircuit, IBMQ

import time

q_system = H2()
hamiltonian = q_system.jw_qubit_ham
n_qubits = q_system.n_qubits

pauli_hf = QasmBackend.ham_expectation_value([], [], q_system=q_system,
                                             n_shots = 100)
ref_hf = q_system.hf_energy
print('pauli_hf = {}'.format(pauli_hf))
print('ref_hf = {}'.format(ref_hf))

# t0 = time.time()
# op = QasmBackend.op_from_ham(hamiltonian, n_qubits)
# t1 = time.time()
# message = 'Time for op_from_ham = {}'.format(t1-t0)
# print(message)
#
# init_state_qasm = QasmUtils.hf_state(q_system.n_electrons)
# header = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{}];\n'.format(n_qubits)
# qasm_psi = header + init_state_qasm
#
# t0 = time.time()
# circ_psi = QuantumCircuit.from_qasm_str(qasm_psi)
# psi = CircuitStateFn(circ_psi)
# t1 = time.time()
# message = 'Time to generate circuitstatefn from qasm_psi = {}'.format(t1-t0)
# print(message)
#
# IBMQ.load_account()
# provider = IBMQ.get_provider(hub='ibm-q')
# backend = provider.get_backend('ibmq_16_melbourne')
# noise_model = NoiseModel.from_backend(backend)
#
# backend = QasmSimulator(method='statevector', noise_model=noise_model)
# q_instance = QuantumInstance(backend, shots=1024)
#
# measurable_expression = StateFn(op, is_measurement=True).compose(psi)
#
# # convert to expectation value
# t0 = time.time()
# expectation = PauliExpectation().convert(measurable_expression)
# t1 = time.time()
# message = 'Time to convert to expectation value = {}'.format(t1-t0)
# print(message)
#
# # get state sampler (you can also pass the backend directly)
# t0 = time.time()
# sampler = CircuitSampler(q_instance).convert(expectation)
# t1 = time.time()
# message = 'Time to get state sampler = {}'.format(t1-t0)
# print(message)
#
#
# print('Sampled:', sampler.eval().real)
# print('hf_energy= {}'.format(q_system.hf_energy))
