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
from src.q_systems import *
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
    r = 1.316
    molecule = BeH2(r=r)  # frozen_els={'occupied': [0, 1], 'unoccupied': []})

    # logging
    LogUtils.log_config()

    # df = pandas.read_csv("../results/iter_vqe_results/vip/LiH_g_adapt_gsdfe_comp_exc_r=3_30-Oct-2020.csv")
    # df = pandas.read_csv("../results/iter_vqe_results/vip/BeH2_h_adapt_gsdqe_comp_pairs_15-Sep-2020.csv")
    df = pandas.read_csv("../results/iter_vqe_results/vip/BeH2_g_adapt_gsdfe_27-Aug-2020.csv")

    state = DataUtils.ansatz_from_data_frame(df, molecule)
    ansatz = state.ansatz_elements
    var_parameters = state.parameters
    ansatz = ansatz

    # var_parameters = list(df['var_parameters'])[:49]
    var_parameters = var_parameters

    global_cache = GlobalCache(molecule)
    global_cache.calculate_exc_gen_sparse_matrices_dict(ansatz)
    global_cache.calculate_commutators_sparse_matrices_dict(ansatz)

    optimizer = 'BFGS'
    optimizer_options = {'gtol': 1e-8}

    print('spagetti')
