from openfermion.hamiltonians import MolecularData
from openfermionpsi4 import run_psi4
from openfermion import QubitOperator
from openfermion.transforms import get_fermion_operator, jordan_wigner, get_sparse_operator
from scipy.linalg import eigh
from openfermion.utils import jw_hartree_fock_state
import scipy
from src.ansatz_elements import UCCSD, DoubleExchange, SingleExchange
import numpy
from src.backends import QiskitSimulation, MatrixCalculation
from src.vqe_runner import VQERunner
from src.molecules import H2, HF
import openfermion
import qiskit
import time

import matplotlib.pyplot as plt

from src.utils import QasmUtils
from src import backends

if __name__ == "__main__":

    angle = - 1

    qasm_1 = ['']
    qasm_1.append(QasmUtils.qasm_header(4))
    # qasm_1.append('x q[3];\n')
    qasm_1.append('h q[1];\n')
    qasm_1.append('h q[2];\n')
    qasm_1.append('x q[0];\n')

    qasm_1.append(SingleExchange(0, 3).get_qasm([angle]))
    statevector_1 = QiskitSimulation.get_statevector_from_qasm(''.join(qasm_1)).round(5)

    print(statevector_1)

    qasm_2 = ['']
    qasm_2.append(QasmUtils.qasm_header(4))
    qasm_2.append('h q[1];\n')
    qasm_2.append('h q[2];\n')
    qasm_2.append('x q[0];\n')

    excitation = UCCSD(4, 2).get_single_excitation_list()[1]
    excitation_qasm = excitation.get_qasm([angle])
    qasm_2.append(excitation_qasm)
    statevector_2 = QiskitSimulation.get_statevector_from_qasm(''.join(qasm_2)).round(10)

    print(statevector_2)

    print('spagetti')

# solution for HF at r =0.995
# excitation_pars = numpy.array([ 3.57987953e-05, -0.00000000e+00, -0.00000000e+00,  3.57756650e-05,2.87818648e-03, -0.00000000e+00, -0.00000000e+00,  2.88683570e-03,1.23046624e-02, -0.00000000e+00, -0.00000000e+00,  1.23512383e-02,-0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,-0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,3.63737395e-04, -0.00000000e+00, -4.74828958e-04, -0.00000000e+00,-7.91240710e-05, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,-0.00000000e+00,  4.74843083e-04, -0.00000000e+00,  7.91811790e-05,-0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,-0.00000000e+00,  2.40169364e-02, -0.00000000e+00, -2.99864664e-02,-0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, 2.99970157e-02, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,  1.35295339e-01, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,  1.77997072e-02, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, 1.78098232e-02])
