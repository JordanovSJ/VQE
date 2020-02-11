# this module contains functions not designated a better place
import scipy
import openfermion
from vqe.utils import QasmUtils


def prepare_statevector_as_matrix(excitations_list, initial_statevector):

    statevector = initial_statevector
    # n_qubits = len(statevector)

    for excitation in excitations_list:
        operator, parameter = excitation

        # operator_matrix = openfermion.transforms.get_sparse_operator(operator, n_qubits)
        statevector = scipy.sparse.linalg.expm_multiply(-1j*parameter*operator, statevector)

    return statevector


# get a list of compressed sparse row matrices, corresponding to the excitation list, including the var. params
def get_excitation_matrix_list(self, params):

        assert len(self.excitation_list) == len(params)

        excitation_matrix_list = []
        for i, excitation in enumerate(self.excitation_list):
            excitation_matrix_list.append(self.get_qubit_operator_exponent_matrix(params[i]*excitation))

        return excitation_matrix_list


def get_excitation_list_qasm(excitation_list, var_parameters, gate_counter):
    qasm = ['']
    # iterate over all excitations (each excitation is represented by a sum of products of pauli operators)
    for i, excitation in enumerate(excitation_list):
        # print('Excitation ', i)  # testing
        # iterate over the terms of each excitation (each term is a product of pauli operators, on different qubits)
        # TODO replace with the function from QasmUtils
        for exponent_term in excitation.terms:
            exponent_angle = var_parameters[i] * excitation.terms[exponent_term]
            assert exponent_angle.real == 0
            exponent_angle = exponent_angle.imag
            qasm.append(QasmUtils.get_exponent_qasm(exponent_term, exponent_angle))

    return ''.join(qasm)