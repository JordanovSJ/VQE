from src.ansatz_element_lists import *
from src.backends import QiskitSim
import qiskit
import time

from src.vqe_runner import *
from src.q_systems import *
from src.adapt_utils import GradAdaptUtils
import numpy, math


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


def find_ys(x):
    ys = []
    for y in range(1,x):
        if math.gcd(x,y) == 1:
            ys.append(y)
    return ys


if __name__ == "__main__":

    Ds = list(numpy.zeros(1001))
    count = 0
    x = 2
    while Ds[1000] == 0:
        if (x / 1000) % 1 == 0:
            print('len D {}'.format(count))
            print('X ', x)

        ys = find_ys(x)
        for y in ys:
            D = (x**2 - 1)/y**2
            if D % 1 == 0 and D <= 1000:
                if Ds[int(D)] == 0:
                    count += 1
                    Ds[int(D)] = x
        x += 1
    print(max(Ds))
    print('spagetti')
