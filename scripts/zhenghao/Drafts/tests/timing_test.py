from src.q_systems import LiH, H2
from scripts.zhenghao.noisy_backends import QasmBackend
from src.ansatz_element_sets import UCCSD
from src.backends import QiskitSimBackend
from src.utils import QasmUtils

import time
import numpy as np

t0 = time.time()

molecule = LiH()
n_qubits= molecule.n_qubits
n_electrons = molecule.n_electrons

hamiltonian = molecule.jw_qubit_ham
ham_terms = hamiltonian.terms
test_term = list(ham_terms.keys())[99]

uccsd = UCCSD(n_qubits, n_electrons)
ansatz = uccsd.get_excitations()

var_pars = np.zeros(len(ansatz))
var_pars += 0.1

qasm_ansatz = QasmBackend.qasm_from_ansatz(ansatz, var_pars)
init_state_qasm = QasmUtils.hf_state(molecule.n_electrons)
qasm_psi = init_state_qasm + qasm_ansatz

expectation_value = QasmBackend.eval_expectation_value(qasm_psi, test_term, n_qubits)
print(expectation_value)

t1 = time.time()
total_time = t1 - t0

print('Time used = {}'.format(total_time))