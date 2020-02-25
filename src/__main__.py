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

    # uccsd = UCCSD(molecule.n_orbitals, molecule.n_electrons)

    ansatz_element_1 = DoubleExchangeAnsatzElement([4, 5], [10, 11])
    ansatz_element_2 = DoubleExchangeAnsatzElement([0, 1], [10, 11])
    ansatz_element_3 = ExchangeAnsatzElement(5, 11)
    ansatz_element_4 = ExchangeAnsatzElement(2, 10)

    ansatz_elements = [ansatz_element_2]#, ansatz_element_2, ansatz_element_3, ansatz_element_2, ansatz_element_4]#, ExchangeAnsatzElement(1, 2), ExchangeAnsatzElement(0, 3)]

    init_var_pars = [0.27533071742668674, 0.10432104737778966, -0.03840191962508642, 0.07229824986909547, -0.009658150286114572]
    vqe_runner = VQERunner(molecule, backend=QiskitSimulation, ansatz_elements=ansatz_elements,
                           molecule_geometry_params={'distance': r}, print_var_parameters=True)#, optimizer='Nelder-Mead')

    t0 = time.time()
    result = vqe_runner.vqe_run(ansatz_elements=ansatz_elements)
    t = time.time()

    logging.critical(result)
    print(result)
    print('TIme for running: ', t - t0)

    print('Pizza')
