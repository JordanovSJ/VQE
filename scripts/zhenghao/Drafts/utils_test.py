from scripts.zhenghao.utils_li import *
from qiskit import QuantumCircuit

qasm_str = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n\nqreg q0[2];\nh q0[0];\ncx q0[0],q0[1];\n'
qasm_str0 = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n\nqreg q1[5];\nh q1[0];\ncx q1[0],q1[1];\nx q1[3];\nz q1[2];\n'
circuit1 = QuantumCircuit.from_qasm_str(qasm_str)
circuit0 = QuantumCircuit.from_qasm_str(qasm_str0)
print(circuit1)
print(circuit0)

qasm_str_append = QasmStrUtils.qasm_append(qasm_str, qasm_str0)
circuit_append = QuantumCircuit.from_qasm_str(qasm_str_append)
print(circuit_append)

# qasm_str_extract, qasm_qreg_test = QasmStrUtils.extract_qreg(qasm_str)
#
# qasm_str_invert = QasmStrUtils.qasm_invert(qasm_str)
#

# circuit2 = QuantumCircuit.from_qasm_str(qasm_str_invert)
# print(circuit2)

# qasm_qreg_list = QasmStrUtils.qreg_declare_list_from_qasm(qasm_str)
# print(qasm_qreg_list)
#
# qasm_qreg_dicts = QasmStrUtils.qreg_name_size_dicts_from_qasm(qasm_str)
# print(qasm_qreg_dicts)
# qreg_size = QasmStrUtils.qreg_size_from_qasm(qasm_str)
# print(qreg_size)

# qasm_str_double = QasmStrUtils.qasm_compose(qasm_str, qasm_str)[0]
# circuit3 = QuantumCircuit.from_qasm_str(qasm_str_double)
# print(circuit3)
#
# qasm_str_double_measure_all = QasmStrUtils.qasm_measure_all(qasm_str_double)
# circuit4 = QuantumCircuit.from_qasm_str(qasm_str_double_measure_all)
# print(circuit4)
#
# qasm_swap = QasmStrUtils.swap_test_qasm(qasm_str, qasm_str)
# circuit_swap = QuantumCircuit.from_qasm_str(qasm_swap)
# print(circuit_swap)