from openfermion.hamiltonians import MolecularData
from openfermionpsi4 import run_psi4
from openfermion import QubitOperator
from openfermion.transforms import get_fermion_operator, jordan_wigner, get_sparse_operator
from scipy.linalg import eigh
from openfermion.utils import jw_hartree_fock_state
import scipy
from src.ansatz_elements import UCCSD, DoubleExchange, SingleExchange, DoubleExcitation, EfficientDoubleExcitation
from src.test_ansatz_elements import EfficientDoubleExchange, EfficientDoubleExcitation2
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

    angle = 0.1
    n = 4
    qasm_init = ['']
    qasm_init.append(QasmUtils.qasm_header(n))
    qasm_init.append('x q[0];\n')
    # qasm_init.append('h q[1];\n')
    qasm_init.append('x q[1];\n')
    # qasm_init.append('x q[2];\n')

    qasm_1 = ['']
    qasm_1 += qasm_init

    # qasm_1.append(DoubleExchange([0, 1], [2, 3], d_exc_correction=False, parity_dependence=False,
    #                              rescaled_parameter=False).get_qasm([angle]))
    qasm_1.append(DoubleExcitation([0, 1], [2, 3]).get_qasm([angle]))

    # qasm_1.append('ry({}) q[{}];\n'.format(numpy.pi / 4, 2))
    # qasm_1.append('cx q[{}], q[{}];\n'.format(1, 2))
    # qasm_1.append('ry({}) q[{}];\n'.format(-numpy.pi / 4, 2))
    #
    # qasm_1.append('cz q[{}], q[{}];\n'.format(0, 2))
    #
    # qasm_1.append('ry({}) q[{}];\n'.format(numpy.pi / 4, 2))
    # qasm_1.append('cx q[{}], q[{}];\n'.format(1, 2))
    # qasm_1.append('ry({}) q[{}];\n'.format(-numpy.pi / 4, 2))

    # qasm_1.append('cz q[{}], q[{}];\n'.format(0, 2))

    statevector_1 = QiskitSimulation.get_statevector_from_qasm(''.join(qasm_1)).round(10)

    print(statevector_1)

    qasm_2 = ['']
    qasm_2 += qasm_init

    # qasm_2.append(EfficientDoubleExchange([0, 1], [2, 3], d_exc_correction=False, parity_dependence=False,
    #                                       rescaled_parameter=False).get_qasm([angle]))
    qasm_2.append(EfficientDoubleExcitation2([0, 1], [2, 3]).get_qasm([angle]))

    statevector_2 = QiskitSimulation.get_statevector_from_qasm(''.join(qasm_2)).round(10)

    print(statevector_2)

    print('spagetti')
