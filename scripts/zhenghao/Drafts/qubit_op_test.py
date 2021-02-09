from src.q_systems import *
from openfermion import QubitOperator
from scripts.zhenghao.utils_li import *
from qiskit import QuantumCircuit
from scripts.zhenghao.noisy_backends import *

system = LiH()
hamiltonian = system.jw_qubit_ham

ham_terms = hamiltonian.terms
ham_terms_keys = list(ham_terms.keys())

op_key = ham_terms_keys[100]

qasm = QubitOpUtils.qasm_from_op_key(op_key)
circ = QuantumCircuit.from_qasm_str(qasm)
print(circ)

