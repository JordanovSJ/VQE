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
    molecule = H2()
    ansatz = GSDExcitations(4, 2, ansatz_element_type='eff_f_exc').get_excitations()

    LogUtils.log_config()

    # var_pars = [ 1.73588403e-32,  3.85808261e-16, -2.40495012e-16,  1.44296833e-16,
    #              1.92369605e-16, -3.37073840e-32,  1.38756291e-06,  1.69449961e-32]

    excitation = EffDFExc([0, 1], [2, 3], system_n_qubits=4)
    H_lower_state_terms = [[1.137, Ansatz([excitation], [0.11176849919227788], 4, 2)]]
    molecule.H_lower_state_terms = H_lower_state_terms

    # optimizer = 'Nelder-Mead'
    # optimizer_options = {'gtol': 1e-8}
    #
    # vqe_runner = VQERunner(molecule, backend=QiskitSim, optimizer=optimizer, optimizer_options=None,
    #                        print_var_parameters=False, use_ansatz_gradient=False)
    #
    # result = vqe_runner.vqe_run(ansatz=ansatz, excited_state=1)

    # result =vqe_runner.get_energy([0.11176849919227788], [EffDFExc([0, 1], [2, 3], 4)], QiskitSim)

    # print(result)

    var_pars = list(numpy.zeros(len(ansatz)))
    var_pars[1] += 0.0
    grad = backends.QiskitSim.ansatz_gradient(var_pars, ansatz, q_system=molecule, excited_state=1)
    print([element.element for element in ansatz])
    print(grad)

    # pars = (numpy.arange(50) - 25)*numpy.pi/50
    # energies = []
    # for par in pars:
    #     print(par)
    #     energy = backends.QiskitSim.ham_expectation_value_exc_state(molecule.jw_qubit_ham, [EffSFExc(0, 2, system_n_qubits=4)], [par],
    #                                                                 q_system=molecule, excited_state=1)
    #     energies.append(energy)
    #
    # plt.plot(pars, energies)
    # plt.show()

    print('spagetti')
