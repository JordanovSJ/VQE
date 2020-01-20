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
from src.vqe_runner import VQERunner
from src.molecules import H2, HF
import openfermion

if __name__ == "__main__":
    molecule = HF
    vqe_runner = VQERunner(molecule, molecule_geometry_params={'distance': 0.995})
    h = vqe_runner.jw_ham_qubit_operator
    excitation_list = UCCSD(molecule.n_orbitals, molecule.n_electrons).get_ansatz_elements()
    excitation_pars = numpy.zeros(len(excitation_list))

    E = QiskitSimulation.get_energy(h, excitation_list, excitation_pars, 12, 10)

    print(E)
    # print(excitation_list)
    print('spagetti')

# solution for HF at r =0.995
# excitation_pars = numpy.array([ 3.57987953e-05, -0.00000000e+00, -0.00000000e+00,  3.57756650e-05,2.87818648e-03, -0.00000000e+00, -0.00000000e+00,  2.88683570e-03,1.23046624e-02, -0.00000000e+00, -0.00000000e+00,  1.23512383e-02,-0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,-0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,3.63737395e-04, -0.00000000e+00, -4.74828958e-04, -0.00000000e+00,-7.91240710e-05, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,-0.00000000e+00,  4.74843083e-04, -0.00000000e+00,  7.91811790e-05,-0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,-0.00000000e+00,  2.40169364e-02, -0.00000000e+00, -2.99864664e-02,-0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, 2.99970157e-02, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,  1.35295339e-01, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,  1.77997072e-02, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, 1.78098232e-02])
