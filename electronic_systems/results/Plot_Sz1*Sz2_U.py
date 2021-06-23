import openfermion

from electronic_systems import ElectronicSystem, ham_14_qubits, ham_16_qubits
import numpy
from src.vqe_runner import VQERunner
from src.ansatz_elements import *
from src.backends import *
from src.utils import LogUtils
from src.cache import *


def ansatz_1():

    n_orbitals = 14
    ansatz = []
    # ansatz.append(Gen2QubitAnsatzElement(0, 1))

    for i in range(n_orbitals):
        for j in range(i + 1, n_orbitals):
            if i % 2 == j % 2:
                ansatz.append(EffSFExc(i, j, system_n_qubits=n_orbitals))

    for i, j, k, l in itertools.combinations(range(n_orbitals), 4):
        # spin conserving excitations only
        if i % 2 + j % 2 == k % 2 + l % 2:
            ansatz.append(DFExc([i, j], [k, l], system_n_qubits=n_orbitals))
            # ansatz.append(DFExc([i, k], [j, l], system_n_qubits=n_orbitals))
            # ansatz.append(DFExc([i, l], [k, j], system_n_qubits=n_orbitals))

    return ansatz


if __name__ == "__main__":
    n_orbitals = 14
    n_electrons = 10

    for U in numpy.linspace(0.1, 0.4, 50):
        H = ham_14_qubits(U)
        e_system = ElectronicSystem(H, n_orbitals, n_electrons)

        LogUtils.log_config()

        ansatz = ansatz_1()

        init_qasm = None

        global_cache = GlobalCache(e_system, excited_state=0)
        global_cache.calculate_exc_gen_sparse_matrices_dict(ansatz)

        backend = MatrixCacheBackend

        optimizer = 'BFGS'
        optimizer_options = {'gtol': 10e-8, 'maxiter': 10}
        vqe_runner = VQERunner(e_system, backend=backend, print_var_parameters=False, use_ansatz_gradient=True,
                           optimizer=optimizer, optimizer_options=optimizer_options)

        result = vqe_runner.vqe_run(ansatz=ansatz,  cache=global_cache, init_state_qasm=init_qasm)

        parameters = result.x
        statevector = global_cache.get_statevector(ansatz, parameters, init_state_qasm=init_qasm)
        statevector = statevector.todense()

        operator = openfermion.FermionOperator('([0^ 0 2^ 2]-[0^ 0 3^ 3]-[1^ 1 2^ 2]+[1^ 1 3^ 3])/4')
        operator = openfermion.jordan_wigner(operator)
        operator = openfermion.get_sparse_operator(operator, n_orbitals)

        expectation_value = statevector.dot(operator).dot(statevector.conj().transpose())

        print(expectation_value)
        print(U)


