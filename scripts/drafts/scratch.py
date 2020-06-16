from src.ansatz_element_lists import *
from src.backends import QiskitSimulation
import qiskit

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
    molecule = HF
    r = 0.995

    uccsd = UCCSD(molecule.n_orbitals, molecule.n_electrons)

    ansatz_elements = uccsd.get_ansatz_elements()
    target_ansatz_element = DoubleQubitExcitation([4,5],[10,11])

    vqe_runner = VQERunner(molecule, backend=QiskitSimulation)

    q_H = vqe_runner.jw_qubit_ham

    backend = backends.QiskitSimulation

    # grad = GradAdaptUtils.get_excitation_energy_gradient(target_ansatz_element, [], [], q_H, 12, 10, backend)
    grads = GradAdaptUtils.get_most_significant_ansatz_element(ansatz_elements, q_H, 12, 10, backend, multithread=False)

    print(grads)
    print('spagetti')
