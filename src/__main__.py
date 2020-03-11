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

    ansatz_element_1 = DoubleExchangeAnsatzElement([4, 5], [10, 11], rescaled=True)
    ansatz_element_2 = DoubleExchangeAnsatzElement([3, 4], [10, 11])#, rescaled=True)
    ansatz_element_3 = ExchangeAnsatzElement(5, 11)
    ansatz_element_4 = ExchangeAnsatzElement(2, 10)
    ansatz_element_5 = DoubleExchangeAnsatzElement([2, 3], [10, 11])#, rescaled=True)

    ansatz_elements = [ansatz_element_1]#, ansatz_element_2, ansatz_element_5, ansatz_element_3, ansatz_element_2]
    init_var_pars = [0.2739346112349019, 0.08593024873566706, 0.10911150073047889, -0.039011893591530344, 0.10037893539114985]

    vqe_runner = VQERunner(molecule, backend=QiskitSimulation, ansatz_elements=ansatz_elements,
                           molecule_geometry_params={'distance': r}, print_var_parameters=True)#, optimizer='Nelder-Mead', optimizer_options={'xatol': 2e-3, 'fatol': 1e-3})

    t0 = time.time()
    result = vqe_runner.vqe_run(ansatz_elements=ansatz_elements)#, initial_var_parameters=init_var_pars)
    t = time.time()

    logging.critical(result)
    print(result)
    print('TIme for running: ', t - t0)

    print('Pizza')
