import numpy
from electronic_systems import ElectronicSystem, ham_14_qubits, ham_16_qubits
import numpy
from src.vqe_runner import VQERunner
from src.ansatz_elements import *
from src.backends import *
from src.utils import LogUtils
from src.cache import *
from src.molecules.molecules import *


def v_n_entropy(statevector, qubits_A):
    qubits_A = set(qubits_A)
    n_qubits = numpy.log2(len(statevector))
    n_qubits_A = len(qubits_A)
    assert n_qubits % 1 == 0
    assert n_qubits_A < n_qubits

    qubits_B = set(numpy.arange(n_qubits)).difference(qubits_A)
    k_s = [0]
    for qubit in qubits_B:
        new_k_s = numpy.array(k_s.copy())
        new_k_s = new_k_s + int(2 ** qubit)
        k_s += list(new_k_s)

    # density matrix for the state of qubits A
    rho_A = numpy.zeros([2**n_qubits_A, 2**n_qubits_A])

    for i in range(2**n_qubits_A):
        for j in range(2**n_qubits_A):
            # MAGIC
            for k in k_s:
                print(i, j, k)
                rho_A[i, j] += statevector[i + k]*statevector[j + k].conjugate()

    entropy = - rho_A.dot(scipy.linalg.logm(rho_A)).trace()

    return entropy, rho_A


if __name__ == "__main__":
    system = H2()
    H_sparse_matrix = get_sparse_operator(system.qubit_ham)
    eigvv = scipy.sparse.linalg.eigs(H_sparse_matrix.todense(), k=10, which='SR')

    eigvectors = eigvv[1].T

    vector = [1/2**0.5, 0, 0, 1/2**0.5]

    print(v_n_entropy(vector, [0])[0])

    print('emi sasho')
