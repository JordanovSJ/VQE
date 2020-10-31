# from src.ansatze import *
from src.backends import QiskitSim
import qiskit
import time
import matplotlib.pyplot as plt

from src.iter_vqe_utils import *
from src.ansatz_elements import *
from src.vqe_runner import *
from src.q_systems import *
from src.backends import  *
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
    df = pandas.read_csv('../../results/iter_vqe_results/BeH2_g_adapt_gsdfe_not_eff_comp_exc_29-Oct-2020.csv')
    cnots = []
    qubitss = df['element_qubits'].values
    for qubits in qubitss:

        if cnots == []:
            previos = 0
        else:
            previos = cnots[-1]

        qubits = ast.literal_eval(qubits)
        spin_comp_qubits = [AnsatzElement.spin_complement_orbitals(qubits[0]),AnsatzElement.spin_complement_orbitals(qubits[1])]
        if len(qubits[0]) == 1:
            cnot_count = 2*abs(qubits[0][0] - qubits[1][0])+1 + previos
            if {qubits[0][0], qubits[1][0]} != {spin_comp_qubits[0][0], spin_comp_qubits[1][0]}:
                cnot_count += 2*abs(spin_comp_qubits[0][0] - spin_comp_qubits[1][0])+1
        else:
            set_qubits = [*qubits[0], *qubits[1]]
            set_qubits.sort()
            cnot_count = 2*(set_qubits[3]-set_qubits[2] + set_qubits[1]-set_qubits[0])+9 + previos

            if [set(qubits[0]), set(qubits[1])] != [set(spin_comp_qubits[0]), set(spin_comp_qubits[1])] and \
               [set(qubits[0]), set(qubits[1])] != [set(spin_comp_qubits[1]), set(spin_comp_qubits[0])]:
                set_qubits = [*spin_comp_qubits[0], *spin_comp_qubits[1]]
                set_qubits.sort()
                cnot_count += 2 * (set_qubits[3] - set_qubits[2] + set_qubits[1] - set_qubits[0]) + 9
        cnots.append(cnot_count)

    df['cnot_count'] = cnots
    df.to_csv('../../results/iter_vqe_results/BeH2_g_adapt_gsdfe_not_eff_comp_exc_29-Oct-2020_corrected.csv')

    # grad = QiskitSim.excitation_gradient(ansatz[0],[],[], molecule)
    # print(result)
    print('spagetti')
