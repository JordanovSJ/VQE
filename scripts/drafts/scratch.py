# from src.ansatze import *
from src.cache import *
from src.backends import QiskitSim
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
    r=1.316
    molecule = BeH2(r=r)
    ansatz = GSDExcitations(molecule.n_qubits, molecule.n_electrons).get_double_excitations()[:10]
    var_parameters = list(numpy.random.random(len(ansatz))/5)
    global_cache = GlobalCache(molecule)
    global_cache.calculate_exc_gen_matrices(ansatz)
    # global_cache.calculate_commutators_matrices(ansatz)
    t0 = time.time()
    statevector1 = QiskitSim.statevector_from_ansatz(ansatz, var_parameters, molecule.n_qubits, molecule.n_electrons)
    print('t1: ', time.time()-t0)
    t0 = time.time()
    statevector2 = MatrixCalculation.statevector_from_ansatz(ansatz, var_parameters, molecule.n_qubits, molecule.n_electrons, global_cache)
    print('t2: ', time.time()-t0)

    print((statevector1-statevector2).dot(statevector1-statevector2))
    # grad = QiskitSim.excitation_gradient(ansatz[0],[],[], molecule)
    # print(result)
    print('spagetti')
