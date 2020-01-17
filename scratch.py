from openfermion.hamiltonians import MolecularData
from openfermionpsi4 import run_psi4
from openfermion import QubitOperator
from openfermion.transforms import get_fermion_operator, jordan_wigner, get_sparse_operator
from scipy.linalg import eigh
from openfermion.utils import jw_hartree_fock_state
import scipy
from src.ansatz_types import UCCSD
import numpy
from src.backends import QiskitSimulation, MatrixCalculation
import openfermion

if __name__ == "__main__":

    angle = numpy.pi/8

    exp_operator_tuple = ((0, 'X'), (1, 'Y'), (2, 'X'))
    # exp_operator_tuple = ((0, 'X'), (1, 'X'))
    qasm = QiskitSimulation.get_qasm_header(3)
    qasm += QiskitSimulation.get_exponent_qasm(exp_operator_tuple, 1j * angle)
    # qasm += 'x q[0];\n'
    qiskit_statevector = QiskitSimulation.get_statevector_from_qasm(qasm)
    qiskit_statevector = qiskit_statevector*numpy.exp(1j * angle)  # correct for global phase
    qiskit_statevector = qiskit_statevector.round(2)  # round for the purpose of testing

    # openfermion has reversed order of qubits?
    exp_operator_qo = 1j * openfermion.QubitOperator('X0 Y1 X2')
    exp_matrix = MatrixCalculation.get_qubit_operator_exponent_matrix(exp_operator_qo, 3, angle).todense()
    array_statevector = numpy.zeros(8)
    array_statevector[0] = 1
    array_statevector = numpy.array(exp_matrix.dot(array_statevector))[0].round(2)

    print(qiskit_statevector)
    print(array_statevector)

    print('spagetti')


