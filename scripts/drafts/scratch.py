from src.ansatz_element_lists import *
from src.backends import QiskitSim
import qiskit
import time

from src.vqe_runner import *
from src.q_systems import *
from src.adapt_utils import GradAdaptUtils


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
    # r = 1.

    uccsd = UCCSD(molecule.n_orbitals, molecule.n_electrons)

    ansatz_elements = uccsd.get_ansatz_elements()
    var_params = list(numpy.zeros(len(ansatz_elements)))

    q_H = molecule.jw_qubit_ham

    backend = backends.QiskitSim

    t0 = time.time()
    print(t0)
    grad_1 = QiskitSim.ansatz_gradient(molecule, ansatz_elements, var_params)
    print(time.time() - t0)

    print(grad_1)
    print('spagetti')
