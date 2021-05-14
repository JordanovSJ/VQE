from electronic_systems import ElectronicSystem, ham_14_qubits, ham_16_qubits
import numpy
from src.vqe_runner import VQERunner
from src.ansatz_elements import *
from src.backends import *
from src.utils import LogUtils
from src.cache import *


class Gen2QubitAnsatzElement(AnsatzElement):
    def __init__(self, qubit_1, qubit_2):
        self.qubits = [qubit_1, qubit_2]
        super(Gen2QubitAnsatzElement, self).\
            __init__(element='gen_2_qubit_{}_{}'.format(qubit_1, qubit_2), n_var_parameters=3)

    def get_qasm(self, var_parameters):
        qasm = ['']
        qasm.append('rz({}) q[{}];\n'.format(numpy.pi/2, self.qubits[0]))
        qasm.append('rz({}) q[{}];\n'.format(numpy.pi/2, self.qubits[1]))
        qasm.append('ry({}) q[{}];\n'.format(numpy.pi/2, self.qubits[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(self.qubits[1], self.qubits[0]))

        # general qubits 1 and 2 rotations
        qasm.append('ry({}) q[{}];\n'.format(var_parameters[0], self.qubits[0]))
        qasm.append('rz({}) q[{}];\n'.format(var_parameters[1], self.qubits[0]))
        qasm.append('ry({}) q[{}];\n'.format(var_parameters[2], self.qubits[1]))

        qasm.append('cx q[{}], q[{}];\n'.format(self.qubits[1], self.qubits[0]))
        qasm.append('ry({}) q[{}];\n'.format(-numpy.pi/2, self.qubits[1]))
        qasm.append('rz({}) q[{}];\n'.format(-numpy.pi/2, self.qubits[0]))
        qasm.append('rz({}) q[{}];\n'.format(-numpy.pi/2, self.qubits[1]))

        return ''.join(qasm)


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


def ansatz_2():
    n_bath = 5
    n_orbitals = 14
    ansatz = []
    ansatz.append(Gen2QubitAnsatzElement(0, 1))

    for i in range(n_bath):
        ansatz.append(EffSFExc(2*i + 4, 2*i + 5, system_n_qubits=n_orbitals))

    return ansatz


if __name__ == "__main__":
    n_orbitals = 14
    n_electrons = 10
    H = ham_14_qubits()
    e_system = ElectronicSystem(H, n_orbitals, n_electrons)

    LogUtils.log_config()

    ansatz = ansatz_1()
    # initial_parameter_values = numpy.zeros(len(ansatz) + 2)

    # initial state with 10 electrons occupying the 10 lowest (spin-)sites
    init_qasm = ''
    for i in range(n_electrons):
        init_qasm += 'x q[{}];\n'.format(i)
    # init_qasm += 'x q[{}];\n'.format(0)
    # init_qasm += 'x q[{}];\n'.format(1)

    # init_qasm = None

    global_cache = GlobalCache(e_system, excited_state=0)
    global_cache.calculate_exc_gen_sparse_matrices_dict(ansatz)

    backend = MatrixCacheBackend

    # vqe_runner = VQERunner(q_system=e_system, backend=QiskitSimBackend, optimizer='Nelder-Mead',
    #                        optimizer_options=None, use_ansatz_gradient=False)

    optimizer = 'BFGS'
    optimizer_options = {'gtol': 10e-8}
    vqe_runner = VQERunner(e_system, backend=backend, print_var_parameters=False, use_ansatz_gradient=True,
                           optimizer=optimizer, optimizer_options=optimizer_options)

    result = vqe_runner.vqe_run(ansatz=ansatz,  cache=global_cache, init_state_qasm=init_qasm)

    print(result)

    print('yolo')

