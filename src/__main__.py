from src.vqe_runner import VQERunner
from src.q_systems import *
from src.ansatze import *
from src.backends import QiskitSimBackend
from src.utils import LogUtils

import logging
import time
import numpy
import pandas
import datetime
import qiskit


if __name__ == "__main__":

    r = 0.735
    frozen_els = None #{'occupied': [0, 1], 'unoccupied': [6, 7]}
    q_system = H2(r=r) #(r=r, frozen_els=frozen_els)

    # logging
    LogUtils.log_config()

    uccsd = UCCSD(q_system.n_orbitals, q_system.n_electrons)
    ansatz = uccsd.get_excitations()

    optimizer = 'BFGS'
    optimizer_options = {'gtol': 10e-8}
    vqe_runner = VQERunner(q_system, backend=QiskitSimBackend, print_var_parameters=False, use_ansatz_gradient=False,
                           optimizer=optimizer, optimizer_options=optimizer_options)

    t0 = time.time()
    result = vqe_runner.vqe_run(ansatz=ansatz)#, initial_var_parameters=var_parameters)
    t = time.time()

    logging.critical(result)
    print(result)
    print('Time for running: ', t - t0)

    print('Pizza')


