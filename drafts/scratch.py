from openfermion.hamiltonians import MolecularData
from openfermionpsi4 import run_psi4
from openfermion import QubitOperator
from openfermion.transforms import get_fermion_operator, jordan_wigner, get_sparse_operator
from scipy.linalg import eigh
from openfermion.utils import jw_hartree_fock_state
import scipy
from src.ansatz_elements import UCCSD
import numpy
from src.backends import QiskitSimulation, MatrixCalculation
from src.vqe_runner import VQERunner
from src.molecules import H2, HF
import openfermion
import qiskit
import time
from src.utils import QasmUtils
from src import backends

from src.ansatz_elements import ExchangeAnsatz2

if __name__ == "__main__":
    qubits = numpy.arange(4)
    angle = numpy.pi/10

    qasm = ['']
    qasm.append(QasmUtils.qasm_header(4))
    qasm.append('x q[0];\n')
    qasm.append('x q[1];\n')
    qasm.append(ExchangeAnsatz2.double_exchange(angle, qubits))
    qasm = ''.join(qasm)
    statevector = backends.QiskitSimulation.get_statevector_from_qasm(qasm)
    print(statevector)
    print(QasmUtils.gate_count(qasm, 4))

    qasm_2 = ['']
    qasm_2.append(QasmUtils.qasm_header(4))
    qasm_2.append('x q[0];\n')
    qasm_2.append('x q[1];\n')
    double_excitation = UCCSD(4, 2).get_double_excitation_list()[0]
    qasm_2.append(double_excitation.get_qasm([angle]))
    qasm_2 = ''.join(qasm_2)
    statevector_2 = backends.QiskitSimulation.get_statevector_from_qasm(qasm_2)
    print(statevector_2.round(3))
    print(QasmUtils.gate_count(qasm_2, 4))

    print('spagetti')

# solution for HF at r =0.995
# excitation_pars = numpy.array([ 3.57987953e-05, -0.00000000e+00, -0.00000000e+00,  3.57756650e-05,2.87818648e-03, -0.00000000e+00, -0.00000000e+00,  2.88683570e-03,1.23046624e-02, -0.00000000e+00, -0.00000000e+00,  1.23512383e-02,-0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,-0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,3.63737395e-04, -0.00000000e+00, -4.74828958e-04, -0.00000000e+00,-7.91240710e-05, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,-0.00000000e+00,  4.74843083e-04, -0.00000000e+00,  7.91811790e-05,-0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,-0.00000000e+00,  2.40169364e-02, -0.00000000e+00, -2.99864664e-02,-0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, 2.99970157e-02, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,  1.35295339e-01, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,  1.77997072e-02, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, 1.78098232e-02])
