# from src.ansatze import *
from src.cache import *
from src.backends import QiskitSimBackend
import qiskit
import time
import matplotlib.pyplot as plt

from src.iter_vqe_utils import *
from src.ansatz_elements import *
from src.ansatz_element_sets import *
from src.vqe_runner import *
from src.molecules.molecules import *
from src.backends import  *
import numpy, math

import pandas
import ast


def get_circuit_matrix(qasm):
    backend = qiskit.Aer.get_backend('unitary_simulator')
    qiskit_circuit = qiskit.QuantumCircuit.from_qasm_str(qasm)
    result = qiskit.execute(qiskit_circuit, backend).result()
    matrix = result.get_unitary(qiskit_circuit, decimals=15)
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
    str_m = str_m.replace('0j', '0')
    str_m = str_m.replace('j', 'I')
    return str_m


if __name__ == "__main__":
    r = 0.735
    q_system = H2(r=r)  # frozen_els={'occupied': [0, 1], 'unoccupied': []})

    ansatz = [DFExc([0, 1], [2, 3], 4, 'bk')]
    print(len(ansatz))
    backend = MatrixCacheBackend
    global_cache = GlobalCache(q_system)
    global_cache.calculate_exc_gen_sparse_matrices_dict(ansatz)
    global_cache.calculate_commutators_sparse_matrices_dict(ansatz)

    print(global_cache.H_sparse_matrix)
    print(global_cache.get_statevector(ansatz, [0]))
    print(global_cache.get_statevector(ansatz, [0.1]))

    optimizer = 'BFGS'
    optimizer_options = {'gtol': 1e-8}

    print('spagetti')
