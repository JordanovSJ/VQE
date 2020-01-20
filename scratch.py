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
from src.molecules import H2
import openfermion

if __name__ == "__main__":
    molecule = H2
    vqe_runner = VQERunner(molecule, molecule_geometry_params={'distance': 0.735})
    h = vqe_runner.jw_ham_qubit_operator
    excitation_list = UCCSD(4, 2).get_excitation_list()
    excitation_pars = numpy.zeros(5)
    excitation_pars[4] = 0.11
    E = QiskitSimulation.get_energy(h, excitation_list, excitation_pars, 4, 2)

    print(E)
    print(excitation_list)
    print('spagetti')


