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


# get a list of compressed sparse row matrices, corresponding to the excitation list, including the var. params
def get_excitation_matrix_list(self, params):

        assert len(self.excitation_list) == len(params)

        excitation_matrix_list = []
        for i, excitation in enumerate(self.excitation_list):
            excitation_matrix_list.append(self.get_qubit_operator_exponent_matrix(params[i]*excitation))

        return excitation_matrix_list


def prepare_statevector_as_qcirq(excitations_list, initial_statevector=[]):

    # TODO: should return qasm qcirq to be executed on quantum simulator or real device

    return 0