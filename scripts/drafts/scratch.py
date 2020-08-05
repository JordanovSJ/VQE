from src.ansatz_element_lists import *
from src.backends import QiskitSim
import qiskit
import time

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

    molecule = H2()  # frozen_els={'occupied': [0, 1], 'unoccupied': []})
    # r = 1.546

    # logging
    LogUtils.log_cofig()

    df = pandas.read_csv("../../results/adapt_vqe_results/vip/LiH_h_adapt_gsdqe_27-Jul-2020.csv")

    init_ansatz_elements = []
    for i in range(len(df)):
        element = df.loc[i]['element']
        element_qubits = df.loc[i]['element_qubits']
        if element[0] == 'e' and element[4] == 's':
            init_ansatz_elements.append(EfficientSingleFermiExcitation(*ast.literal_eval(element_qubits)))
        elif element[0] == 'e' and element[4] == 'd':
            init_ansatz_elements.append(EfficientDoubleFermiExcitation(*ast.literal_eval(element_qubits)))
        elif element[0] == 's' and element[2] == 'q':
            init_ansatz_elements.append(SingleQubitExcitation(*ast.literal_eval(element_qubits)))
        elif element[0] == 'd' and element[2] == 'q':
            init_ansatz_elements.append(DoubleQubitExcitation(*ast.literal_eval(element_qubits)))
        else:
            print(element, element_qubits)
            raise Exception('Unrecognized ansatz element.')
    # for i in range(len(df)):
    #     excitation = QubitOperator(df.loc[i]['element'])
    #     init_ansatz_elements.append(PauliWordExcitation(excitation))

    init_ansatz_elements = UCCSD(4, 2).get_ansatz_elements()

    init_var_parameters = [0,0,0,0,0] #list(df['var_parameters'])

    t0 = time.time()
    grad = QiskitSim.ansatz_gradient(init_var_parameters, molecule, init_ansatz_elements)
    print(time.time() - t0)
    print(grad)

    print('spagetti')
