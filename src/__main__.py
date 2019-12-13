from src.VQERunner import VQERunner
from src.Molecules import H2, LiH
import logging

if __name__ == "__main__":

    molecule = H2
    vqe_runner = VQERunner(molecule, molecule_geometry_params={'distance': 0.75})

    logging.info('hello there')

    print(vqe_runner.E)

    # TODO add stuff
    print('Pizza')
