from src.vqe_runner import VQERunner
from src.molecules import H2, LiH, HF
from src.ansatz_elements import UCCGSD, UCCSD, DoubleExchangeAnsatzElement
from src.backends import QiskitSimulation
from src.utils import LogUtils

import logging
import time
import numpy
import pandas
import datetime
import qiskit


if __name__ == "__main__":

    molecule = HF
    r = 0.995

    # logging
    LogUtils.log_cofig()

    # uccsd = UCCSD(molecule.n_orbitals, molecule.n_electrons)
    ansatz_element_1 = DoubleExchangeAnsatzElement([4, 5], [10, 11])
    ansatz_element_2 = DoubleExchangeAnsatzElement([2, 3], [10, 11]) #.59660
    ansatz_element_3 = DoubleExchangeAnsatzElement([8, 9], [10, 11])

    # ansatz_element_1 = DoubleExchangeAnsatzElement([0, 1], [2, 3])
    # ansatz_element_1 = ExchangeAnsatz1(molecule.n_orbitals, molecule.n_electrons, n_blocks=2)
    ansatz_elements = [ansatz_element_1, ansatz_element_2, ansatz_element_3]

    vqe_runner = VQERunner(molecule, backend=QiskitSimulation, ansatz_elements=ansatz_elements,
                           molecule_geometry_params={'distance': r}, print_var_parameters=True)#, optimizer='Nelder-Mead')

    t0 = time.time()
    result = vqe_runner.vqe_run()
    t = time.time()

    logging.critical(result)
    print(result)
    print('TIme for running: ', t - t0)

    print('Pizza')
