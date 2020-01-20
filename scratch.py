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
    excitation_list = UCCSD(molecule.n_orbitals, molecule.n_electrons).get_excitation_list()
    excitation_pars = numpy.zeros(len(excitation_list))
    excitation_pars[4] = 0.11
    E = QiskitSimulation.get_energy(h, excitation_list, excitation_pars, 4, 2)

    print(E)
    print(excitation_list)
    print('spagetti')

# excitation_pars = numpy.array([-0.0007274747909297266, 0.0, 0.0009496579162920387, 0.0, 0.00015824814201602467, 0.0, 0.0, 0.0, 0.0, -0.0009496861661300129, 0.0, -0.0001583623579869896, 0.0, 0.0, 0.0, 0.0, 0.0, -0.048033872744446565, 0.0, 0.05997293276886616, 0.0, 0.0, 0.0, 0.0, -0.05999403140234816, 0.0, 0.0, 0.0, 0.0, 0.0, -0.27059067754302085, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.035599414362302254, 0.0, 0.0, 0.0, 0.0, -0.035619646459248006, -7.159759060602119e-05, 0.0, 0.0, -7.155133008015064e-05, -0.005756372964186744, 0.0, 0.0, -0.005773671395555684, -0.024609324739261564, 0.0, 0.0, -0.02470247651690094, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])[::-1]/(-2)
