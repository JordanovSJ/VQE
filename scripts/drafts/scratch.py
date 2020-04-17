from openfermion.hamiltonians import MolecularData
from openfermionpsi4 import run_psi4
from openfermion import QubitOperator
from openfermion.transforms import get_fermion_operator, jordan_wigner, get_sparse_operator
from scipy.linalg import eigh
from openfermion.utils import jw_hartree_fock_state
import scipy
from src.ansatz_elements import UCCSD, DoubleExchange, SingleExchange, DoubleExcitation, EfficientDoubleExcitation
from src.test_ansatz_elements import EfficientDoubleExchange
import numpy
from src.backends import QiskitSimulation, MatrixCalculation
from src.vqe_runner import VQERunner
from src.molecules import H2, HF
import openfermion
import qiskit
import time

import matplotlib.pyplot as plt

from src.utils import QasmUtils
from src import backends

if __name__ == "__main__":

    angle = + 0.05
    n = 4
    qasm_init = ['']
    qasm_init.append(QasmUtils.qasm_header(n))
    qasm_init.append('x q[2];\n')
    # qasm_init.append('h q[1];\n')
    qasm_init.append('x q[3];\n')

    qasm_1 = ['']
    qasm_1 += qasm_init
    # qasm_1.append('x q[0];\n')
    # qasm_1.append('x q[1];\n')
    # qasm_1.append('h q[3];\n')
    # qasm_1.append('h q[2];\n')
    # qasm_1.append('h q[2];\n')
    # qasm_1.append('h q[3];\n')
    # qasm_1.append('h q[4];\n')
    # qasm_1.append('h q[5];\n')

    qasm_1.append(DoubleExchange([0, 1], [2, 3], d_exc_correction=True, parity_dependence=True,
                                 rescaled_parameter=True).get_qasm([angle]))

    statevector_1 = QiskitSimulation.get_statevector_from_qasm(''.join(qasm_1)).round(5)

    print(statevector_1)

    qasm_2 = ['']
    qasm_2 += qasm_init

    qasm_2.append(EfficientDoubleExchange([0, 1], [2, 3], d_exc_correction=True, parity_dependence=True,
                                          rescaled_parameter=True).get_qasm([angle]))
    statevector_2 = QiskitSimulation.get_statevector_from_qasm(''.join(qasm_2)).round(10)

    print(statevector_2)

    print('spagetti')
