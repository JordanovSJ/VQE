from src.vqe_runner import VQERunner
from src.molecules import H2, LiH, HF
from src.ansatz_elements import UCCGSD, UCCSD, DoubleExchangeAnsatzElement, ExchangeAnsatzElement, ESD, EGSD, ExchangeAnsatzBlock
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

    uccsd = UCCSD(molecule.n_orbitals, molecule.n_electrons)
    # ansatz_element_1 = uccsd.get_double_excitation_list()[0]
    ansatz_element_1 = DoubleExchangeAnsatzElement([4, 5], [10, 11]) #.59660

    # ansatz_element_4 = ExchangeAnsatzElement(0, 11)
    # ansatz_element_2 = DoubleExchangeAnsatzElement([2, 3], [10, 11])

    # ansatz_element_1 = DoubleExchangeAnsatzElement([0, 1], [2, 3])
    ansatz_elements = [ansatz_element_1]#, ExchangeAnsatzElement(1, 2), ExchangeAnsatzElement(0, 3)]

    vqe_runner = VQERunner(molecule, backend=QiskitSimulation, ansatz_elements=ansatz_elements,
                           molecule_geometry_params={'distance': r}, print_var_parameters=True)#, optimizer='Nelder-Mead')

    t0 = time.time()
    result = vqe_runner.vqe_run()
    t = time.time()

    logging.critical(result)
    print(result)
    print('TIme for running: ', t - t0)

    print('Pizza')
