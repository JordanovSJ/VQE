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

from src.ansatz_elements import ExchangeAnsatz1

if __name__ == "__main__":
    molecule = H2
    r = 0.735
    max_n_iterations = 2000

    uccsd = UCCSD(molecule.n_orbitals, molecule.n_electrons)
    ansatz_element = ExchangeAnsatz1(molecule.n_orbitals, molecule.n_electrons)
    ansatz_elements = [ansatz_element]

    vqe_runner = VQERunner(molecule, backend=QiskitSimulation, ansatz_elements=ansatz_elements,
                           molecule_geometry_params={'distance': r}, optimizer='Nelder-Mead')

    var_pars = [0.5, 0.2, 0, 0.2]

    E = vqe_runner.get_energy(var_pars, ansatz_elements)

    print(E)
    print('spagetti')

# solution for HF at r =0.995
# excitation_pars = numpy.array([ 3.57987953e-05, -0.00000000e+00, -0.00000000e+00,  3.57756650e-05,2.87818648e-03, -0.00000000e+00, -0.00000000e+00,  2.88683570e-03,1.23046624e-02, -0.00000000e+00, -0.00000000e+00,  1.23512383e-02,-0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,-0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,3.63737395e-04, -0.00000000e+00, -4.74828958e-04, -0.00000000e+00,-7.91240710e-05, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,-0.00000000e+00,  4.74843083e-04, -0.00000000e+00,  7.91811790e-05,-0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,-0.00000000e+00,  2.40169364e-02, -0.00000000e+00, -2.99864664e-02,-0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, 2.99970157e-02, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,  1.35295339e-01, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,  1.77997072e-02, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, 1.78098232e-02])
