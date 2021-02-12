from openfermion.transforms import get_fermion_operator, jordan_wigner, get_sparse_operator
from openfermion import QubitOperator

import qiskit
from qiskit import QuantumCircuit
from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.tools.visualization import plot_histogram

from src.q_systems import LiH, H2
from scripts.zhenghao.noisy_backends import QasmBackend
from src.utils import QasmUtils
from src.ansatz_element_sets import UCCSD
from src.backends import QiskitSimBackend

import numpy as np
import scipy
from functools import partial

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
expectation_value_1 = QasmBackend.ham_expectation_value(var_pars, ansatz, molecule, n_shots=10000)

print(expectation_value_0)
print(expectation_value_1)
