from src.VQERunner import VQERunner
from src.Molecules import H2, LiH
import logging

if __name__ == "__main__":

    molecule = H2
    max_n_interations = 500

    vqe_runner = VQERunner(molecule, molecule_geometry_params={'distance': 0.735})

    result = vqe_runner.vqe_run(max_n_interations)

    print(result)

    # TODO add stuff
    print('Pizza')
