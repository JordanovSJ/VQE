from vqe.vqe import VQERunner
from vqe.molecules import H2, LiH, HF
from vqe.ansatz_elements import UCCGSD, UCCSD, FixedAnsatz1
from vqe.backends import QiskitSimulation
from vqe.utils import LogUtils

import logging
import time
import numpy
import pandas
import datetime
import qiskit


if __name__ == "__main__":

    molecule = H2
    r = 0.735
    max_n_iterations = 2000

    # logging
    LogUtils.log_cofig()

    ansatz_elements = UCCSD(molecule.n_orbitals, molecule.n_electrons).get_ansatz_elements()
    # ansatz_elements = FixedAnsatz1(molecule.n_orbitals, molecule.n_electrons).get_ansatz_elements()

    vqe_runner = VQERunner(molecule, backend=QiskitSimulation, ansatz_elements=ansatz_elements,
                           molecule_geometry_params={'distance': r})
    t0 = time.time()
    result = vqe_runner.vqe_run(max_n_iterations=max_n_iterations)
    t = time.time()

    logging.critical(result)
    print(result)
    print('TIme for running: ', t - t0)

    print('Pizza')
