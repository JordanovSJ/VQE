from src.ansatz_element_lists import *
from src.backends import QiskitSimulation
import qiskit

from src.utils import QasmUtils
from src.vqe_runner import *
from src.molecules import *


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
    molecule = H2
    r = 0.735

    uccsd = UCCSD(molecule.n_orbitals, molecule.n_electrons)

    ansatz_elements = uccsd.get_ansatz_elements()
    target_ansatz_element = ansatz_elements[0]

    vqe_runner = VQERunner(molecule, backend=QiskitSimulation)

    q_H = vqe_runner.jw_qubit_ham

    if target_ansatz_element.order == 1:
        fermi_operator = FermionOperator('[{1}^ {0}] - [{0}^ {1}]'.format(target_ansatz_element.qubit_2, target_ansatz_element.qubit_1))
        exponent_term = jordan_wigner(fermi_operator)
    elif target_ansatz_element.order == 2:
        fermi_operator = FermionOperator('[{2}^ {3}^ {0} {1}] - [{0}^ {1}^ {2} {3}]'
                                         .format(*target_ansatz_element.qubit_pair_1,
                                                 *target_ansatz_element.qubit_pair_2))
        exponent_term = jordan_wigner(fermi_operator)
    else:
        raise Exception('Invalid ansatz element.')

    gradient = backends.QiskitSimulation.get_exponent_energy_gradient(q_H, exponent_term, [], [], 4, 2)
    print(gradient)
    print('spagetti')
