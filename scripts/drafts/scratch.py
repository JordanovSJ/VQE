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
    molecule = BeH2()
    df = pandas.read_csv("../../results/iter_vqe_results/vip/BeH2_g_adapt_gsdfe_27-Aug-2020.csv")
    n_cnot_counts = []

    for i in range(len(df)):
        print(i)
        element = df.loc[i]['element']
        element_qubits = ast.literal_eval(df.loc[i]['element_qubits'])

        if element[0] == 's':
            assert len(element_qubits) == 2
            n_cnots = 3 + 2*(element_qubits[1] - element_qubits[0])
            if i == 0:
                n_cnot_counts = [n_cnots]
            else:
                n_cnot_counts.append(n_cnot_counts[-1] + n_cnots)
        elif element[0] == 'd':
            assert len(element_qubits) == 2
            n_cnots = 13 + 2*(element_qubits[1][1] - element_qubits[1][0] + element_qubits[0][1] - element_qubits[0][0]-2)

            if i == 0:
                n_cnot_counts = [n_cnots]
            else:
                n_cnot_counts.append(n_cnot_counts[-1] + n_cnots)

        else:
            print(element, element_qubits)
            raise Exception('Unrecognized ansatz element.')
    df['cnot_count'] = n_cnot_counts
    df.to_csv("../../results/adapt_vqe_results/vip/BeH2_g_adapt_gsdefe_corrected_09-Sep-2020.csv")

    print('spagetti')
