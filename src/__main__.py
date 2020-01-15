from src.VQERunner import VQERunner
from src.Molecules import H2, LiH, HF
from src.AnsatzType import UCCGSD

import logging

if __name__ == "__main__":

    molecule = HF
    max_n_interations = 500

    vqe_runner = VQERunner(molecule, molecule_geometry_params={'distance': 0.995}) #,excitation_list=UCCGSD(molecule.n_orbitals, molecule.n_electrons).get_excitation_list())

    result = vqe_runner.vqe_run(max_n_interations)

    print(result)

    print('Pizza')
