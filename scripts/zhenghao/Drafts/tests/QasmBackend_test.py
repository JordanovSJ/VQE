from src.q_systems import LiH, H2
from scripts.zhenghao.noisy_backends import QasmBackend
from src.ansatz_element_sets import UCCSD
from src.backends import QiskitSimBackend

import numpy as np
# qasm_test = '\ny q[0];\nz q[1];\nz q[2];\ny q[3];\nx q[4];\nx q[5];\n'

molecule = LiH()
n_qubits= molecule.n_qubits
n_electrons = molecule.n_electrons

hamiltonian = molecule.jw_qubit_ham

uccsd = UCCSD(n_qubits, n_electrons)
ansatz = uccsd.get_excitations()

var_pars = np.zeros(len(ansatz))
var_pars += 0.1

expectation_value_0 = QiskitSimBackend.ham_expectation_value(var_pars, ansatz, molecule)
print(expectation_value_0)

expectation_value_1 = QasmBackend.ham_expectation_value(var_pars, ansatz, molecule)

print(expectation_value_1)
