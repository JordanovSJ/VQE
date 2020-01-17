from src.VQERunner import VQERunner
from src.molecules import H2, LiH, HF
from src.ansatz_types import UCCGSD
from src.backends import QiskitSimulation
import logging

if __name__ == "__main__":

    molecule = H2
    max_n_interations = 500

    # backend=QiskitSimulation, # not working
    vqe_runner = VQERunner(molecule, molecule_geometry_params={'distance': 0.735}) #,excitation_list=UCCGSD(molecule.n_orbitals, molecule.n_electrons).get_excitation_list())

    result = vqe_runner.vqe_run(max_n_interations)

    print(result)

    print('Pizza')
