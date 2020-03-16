from openfermion.hamiltonians import MolecularData
from openfermionpsi4 import run_psi4
from openfermion import QubitOperator
from openfermion.transforms import get_fermion_operator, jordan_wigner, get_sparse_operator
from scipy.linalg import eigh
from openfermion.utils import jw_hartree_fock_state
import scipy
from src.ansatz_elements import UCCSD, DoubleExchangeAnsatzElement, ExchangeAnsatzElement
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

    angle = - 0.1

    ansatz_element_1 = DoubleExchangeAnsatzElement([2, 5], [10, 11], rescaled=True)
    ansatz_element_2 = UCCSD(12, 10).get_double_excitation_list()[19]

    statevetor_1 = QiskitSimulation.get_statevector_from_ansatz_elements([ansatz_element_1], [angle], 12, 10)[0].round(5)
    statevetor_2 = QiskitSimulation.get_statevector_from_ansatz_elements([ansatz_element_2], [angle], 12, 10)[0].round(5)

    for i in range(len(statevetor_1)):
        if statevetor_1[i] != 0 or statevetor_2[i] != 0:
            print('1', i, statevetor_1[i])
            print('2', i, statevetor_2[i])

    print('spagetti')

# solution for HF at r =0.995
# excitation_pars = numpy.array([ 3.57987953e-05, -0.00000000e+00, -0.00000000e+00,  3.57756650e-05,2.87818648e-03, -0.00000000e+00, -0.00000000e+00,  2.88683570e-03,1.23046624e-02, -0.00000000e+00, -0.00000000e+00,  1.23512383e-02,-0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,-0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,3.63737395e-04, -0.00000000e+00, -4.74828958e-04, -0.00000000e+00,-7.91240710e-05, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,-0.00000000e+00,  4.74843083e-04, -0.00000000e+00,  7.91811790e-05,-0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,-0.00000000e+00,  2.40169364e-02, -0.00000000e+00, -2.99864664e-02,-0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, 2.99970157e-02, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,  1.35295339e-01, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,  1.77997072e-02, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, 1.78098232e-02])
