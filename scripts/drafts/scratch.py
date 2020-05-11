from openfermion.hamiltonians import MolecularData
from openfermionpsi4 import run_psi4
from openfermion import QubitOperator
from openfermion.transforms import get_fermion_operator, jordan_wigner, get_sparse_operator
from scipy.linalg import eigh
from openfermion.utils import jw_hartree_fock_state
import scipy
from src.ansatz_element_lists import UCCSD, DoubleExchange, SingleBosExcitation, DoubleFermiExcitation, DoubleBosExcitation
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


def get_circuit_matrix(qasm):
    backend = qiskit.Aer.get_backend('unitary_simulator')
    qiskit_circuit = qiskit.QuantumCircuit.from_qasm_str(qasm)
    result = qiskit.execute(qiskit_circuit, backend).result()
    matrix = result.get_unitary(qiskit_circuit, decimals=5)
    return matrix


def matrix_to_str(matrix):
    str_m = '{'

    for row in matrix:
        str_m += '{'
        for element in row:
            str_m += str(element)
            str_m += ','

        str_m = str_m[:-1]  # remove last coma
        str_m += '},'

    str_m = str_m[:-1]  # remove last coma
    str_m += '}'
    str_m.replace('j', 'I')
    return str_m


if __name__ == "__main__":

    angle = 0.1
    n = 2
    qasm_init = ['']
    qasm_init.append(QasmUtils.qasm_header(n))
    # qasm_init.append('x q[1];\n')
    # qasm_init.append('x q[0];\n')
    # qasm_init.append('h q[1];\n')
    # qasm_init.append('x q[3];\n')
    # qasm_init.append('x q[2];\n')

    # qasm_1 = ['']
    # qasm_1 += qasm_init
    #
    # # qasm_1.append(DoubleExchange([0, 1], [2, 3], d_exc_correction=False, parity_dependence=False,
    # #                              rescaled_parameter=False).get_qasm([angle]))
    # qasm_1.append(DoubleExcitation([0, 1], [2, 3]).get_qasm([angle]))
    #
    # statevector_1 = QiskitSimulation.get_statevector_from_qasm(''.join(qasm_1)).round(10)
    #
    # print(statevector_1)

    qasm_2 = ['']
    qasm_2 += qasm_init

    # qasm_2.append(EfficientDoubleExchange([0, 1], [2, 3], d_exc_correction=False, parity_dependence=False,
    #                                       rescaled_parameter=False).get_qasm([angle]))
    # qasm_2.append(EfficientDoubleExcitation2([0, 1], [2, 3]).get_qasm([angle]))

    # qasm_2.append(QasmUtils.partial_exchange(angle, 0, 1))
    # qasm_2.append('rx({}) q[0];\n'.format(numpy.pi/4))
    # qasm_2.append('cz q[1], q[0];\n')
    # qasm_2.append('rx({}) q[0];\n'.format(-numpy.pi/4))
    # qasm_2.append('cz q[2], q[0];\n')
    # qasm_2.append('rx({}) q[0];\n'.format(numpy.pi/4))
    # qasm_2.append('cz q[1], q[0];\n')
    # qasm_2.append('rx ({}) q[0];\n'.format(-numpy.pi/4))
    # qasm_2.append('cz q[2], q[0];\n')

    qasm_2.append(QasmUtils.controlled_xz(1, 0))

    statevector_2 = QiskitSimulation.get_statevector_from_qasm(''.join(qasm_2)).round(10)

    print(statevector_2)

    print(matrix_to_str(get_circuit_matrix(''.join(qasm_2))))

    print('spagetti')
