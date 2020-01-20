from src.vqe_runner import VQERunner
from src.molecules import H2, LiH, HF
from src.ansatz_types import UCCGSD, UCCSD
from src.backends import QiskitSimulation
import logging
import time

if __name__ == "__main__":

    molecule = H2
    max_n_iterations = 500

    excitation_list = UCCSD(molecule.n_orbitals, molecule.n_electrons).get_excitation_list()

    # backend=QiskitSimulation, # not working
    vqe_runner = VQERunner(molecule, backend=QiskitSimulation, excitation_list=excitation_list, molecule_geometry_params={'distance': 0.735}) #,excitation_list=UCCGSD(molecule.n_orbitals, molecule.n_electrons).get_excitation_list())

    t0 = time.time()
    result = vqe_runner.vqe_run(max_n_iterations)
    t = time.time()

    print(result)
    print('TIme for running: ', t - t0)

    print('Pizza')
