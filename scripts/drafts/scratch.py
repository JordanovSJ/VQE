from src.ansatze import *
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
    molecule = H4(r=0.735)
    # ansatz = GSDExcitations(4, 2, ansatz_element_type='eff_f_exc').get_excitations()
    ansatz = [DQExc([1,3], [5,7], 12)]
    LogUtils.log_config()

    optimizer = 'BFGS'
    optimizer_options = {'gtol': 1e-07}
    vqe_runner = VQERunner(molecule, backend=QiskitSim, optimizer=optimizer, optimizer_options=optimizer_options,
                           use_ansatz_gradient=True, print_var_parameters=True)

    result = vqe_runner.vqe_run(ansatz)

    # grad = QiskitSim.excitation_gradient(ansatz[0],[],[], molecule)
    print(result)
    print('spagetti')
