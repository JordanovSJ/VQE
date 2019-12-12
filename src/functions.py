# this module contains functions not designated a better place
import scipy
import openfermion


def get_double_excitations():
    # TODO
    return 0


def get_single_excitations():
    # TODO
    return 0


def get_qubit_operator_for_excitation():
    # TODO needed?
    return 0


def prepare_statevector_as_matrix(excitations_list, initial_statevector):

    statevector = initial_statevector
    # n_qubits = len(statevector)

    for excitation in excitations_list:
        operator, parameter = excitation

        # operator_matrix = openfermion.transforms.get_sparse_operator(operator, n_qubits)
        statevector = scipy.sparse.linalg.expm_multiply(-1j*parameter*operator, statevector)

    return statevector


def prepare_statevector_as_qcirq(excitations_list, initial_statevector=[]):

    # TODO: should return qasm qcirq to be executed on quantum simulator or real device

    return 0