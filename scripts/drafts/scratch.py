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
    # molecule = BeH2(r=r)
    # ansatz = GSDExcitations(molecule.n_qubits, molecule.n_electrons).get_double_excitations()[:10]
    # var_parameters = list(numpy.random.random(len(ansatz))/5)
    # global_cache = GlobalCache(molecule)
    # global_cache.calculate_exc_gen_matrices(ansatz)

    # global_cache.calculate_commutators_matrices(ansatz)
    # t0 = time.time()
    # statevector1 = QiskitSim.statevector_from_ansatz(ansatz, var_parameters, molecule.n_qubits, molecule.n_electrons)
    # print('t1: ', time.time()-t0)
    # t0 = time.time()
    # statevector2 = ExcStateSim.statevector_from_ansatz(ansatz, var_parameters, molecule.n_qubits, molecule.n_electrons, global_cache)
    # print('t2: ', time.time()-t0)

    # print((statevector1-statevector2).dot(statevector1-statevector2))
    # grad = QiskitSim.excitation_gradient(ansatz[0],[],[], molecule)
    # print(result)

    parameter = 0.1
    n_qubits = 6
    n_electrons = 3
    hf_statevector = numpy.zeros(2 ** n_qubits)
    # MAGIC
    hf_term = 0
    for i in range(n_electrons):
        hf_term += 2 ** (n_qubits - 1 - i)
    hf_statevector[hf_term] = 1
    identity = scipy.sparse.identity(2 ** n_qubits)

    excitation = DFExc([0, 2], [3, 5], n_qubits)
    # excitation = SpinCompDFExc([0, 1], [2, 3], n_qubits)
    excitation_gen_matrix = get_sparse_operator(excitation.excitations_generators, n_qubits)
    excitation_matrix_1 = scipy.sparse.linalg.expm(parameter*excitation_gen_matrix)

    term1 = numpy.sin(parameter)*excitation_gen_matrix
    term2 = (1 - numpy.cos(parameter)) * excitation_gen_matrix*excitation_gen_matrix
    excitation_matrix_2 = identity + term1 + term2

    excitation_matrix_0 = get_circuit_matrix(QasmUtils.qasm_header(n_qubits) + SpinCompDFExc([5, 3], [2, 0], n_qubits).get_qasm([parameter]))

    statevector_0 = QiskitSim.statevector_from_ansatz([excitation], [parameter], n_qubits, n_electrons).round(10)
    statevector_1 = excitation_matrix_1.dot(scipy.sparse.csr_matrix(hf_statevector).transpose().conj()).\
        transpose().conj().todense().round(10)

    statevector_2 = excitation_matrix_2.dot(scipy.sparse.csr_matrix(hf_statevector).transpose().conj()).\
        transpose().conj().todense().round(10)

    print(statevector_0)
    print(statevector_1)
    print(statevector_2)

    m1 = matrix_to_str(numpy.array(excitation_matrix_1.todense()))
    m2 = matrix_to_str(numpy.array(excitation_matrix_2.todense()))
    #
    # m = matrix_to_str(numpy.array((excitation_matrix_1-excitation_matrix_2).todense()))
    # print(m1)
    # print(m2)
    # print(m)


    print('spagetti')
