from src.ansatz_element_lists import *
from src.backends import QiskitSim
import qiskit
import time

from src.ansatz_elements import *
from src.vqe_runner import *
from src.q_systems import *
from src.backends import  *
from src.adapt_utils import GradAdaptUtils
import numpy, math

import pandas
import ast


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

    qasm_1 = ['']
    qasm_1.append(QasmUtils.qasm_header(6))

    # qasm_1.append('x q[0];\n')
    # qasm_1.append('x q[1];\n')
    # # qasm_1.append('h q[3];\n')
    # # qasm_1.append('h q[2];\n')
    # # qasm_1.append('h q[2];\n')
    # # qasm_1.append('h q[3];\n')
    # # qasm_1.append('h q[4];\n')
    # # qasm_1.append('h q[5];\n')
    #
    # # qasm_1.append(QasmUtils.partial_exchange_gate(angle, 0, 2))
    # # qasm_1.append(QasmUtils.partial_exchange_gate(angle,  1, 3))
    # # qasm_1.append('cz q[{}], q[{}];\n'.format(2, 3))
    # # qasm_1.append(QasmUtils.partial_exchange_gate(-angle, 0, 2))
    # # qasm_1.append(QasmUtils.partial_exchange_gate(-angle, 1, 3))
    #
    # qasm_1.append(EffDFExcitation([0, 1], [2, 3]).get_qasm([angle]))
    # statevector_1 = QiskitSim.statevector_from_qasm(''.join(qasm_1)).round(5)
    #
    # print(statevector_1)

    qasm_2 = ['']
    qasm_2.append(QasmUtils.qasm_header(6))
    # qasm_2.append('x q[0];\n')
    # qasm_2.append('x q[1];\n')
    # qasm_2.append('h q[3];\n')
    # qasm_2.append('h q[2];\n')
    # qasm_2.append('h q[2];\n')
    # qasm_2.append('h q[3];\n')
    # qasm_2.append('h q[4];\n')
    # qasm_2.append('h q[5];\n')

    qasm_2.append(EffDFExcitation([0, 3], [5, 4]).get_qasm([-angle]))
    statevector_2 = QiskitSim.statevector_from_qasm(''.join(qasm_2)).round(10)

    print(statevector_2)

    print(matrix_to_str(get_circuit_matrix(''.join(qasm_2))))

    print('spagetti')
