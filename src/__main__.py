from src.vqe_runner import VQERunner
from src.molecules import H2, LiH, HF, BeH2
from src.ansatz_elements import UCCGSD, UCCSD, DoubleExchange, SingleExchange, ESD, EGSD
from src.backends import QiskitSimulation
from src.utils import LogUtils

import logging
import time
import numpy
import pandas
import datetime
import qiskit


if __name__ == "__main__":

    molecule = H2
    r = 0.735

    # logging
    LogUtils.log_cofig()

    uccsd = UCCSD(molecule.n_orbitals, molecule.n_electrons)

    ansatz_elements = uccsd.get_ansatz_elements()

    vqe_runner = VQERunner(molecule, backend=QiskitSimulation, ansatz_elements=ansatz_elements,
                           molecule_geometry_params={'distance': r}, print_var_parameters=True)#, optimizer='Nelder-Mead', optimizer_options={'xatol': 2e-3, 'fatol': 1e-3})

    t0 = time.time()
    result = vqe_runner.vqe_run()#, initial_var_parameters=init_var_pars)
    t = time.time()

    logging.critical(result)
    print(result)
    print('TIme for running: ', t - t0)

    print('Pizza')
