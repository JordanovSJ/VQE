from src.vqe_runner import VQERunner
from src.q_system import *
from src.ansatz_element_sets import *
from src.backends import QiskitSimBackend, MatrixCacheBackend
from src.utils import LogUtils
from src.cache import *

from src.molecules.molecules import *

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

    # uccsd = UCCSD(q_system.n_orbitals, q_system.n_electrons)
    # ansatz = uccsd
    ansatz = UCCSDExcitations(q_system.n_orbitals, q_system.n_electrons, 'q_exc').get_excitations()
    print(len(ansatz))
    backend = MatrixCacheBackend
    global_cache = GlobalCache(q_system)
    global_cache.calculate_exc_gen_sparse_matrices_dict(ansatz)
    # global_cache.calculate_commutators_sparse_matrices_dict()

    # backend = QiskitSimBackend
    # global_cache = None

    optimizer = 'COBYLA'
    optimizer_options = {'gtol': 10e-8}
    vqe_runner = VQERunner(q_system, backend=backend, print_var_parameters=False, use_ansatz_gradient=True,
                           optimizer=optimizer, optimizer_options=optimizer_options)

    t0 = time.time()
    result = vqe_runner.vqe_run(ansatz=ansatz, cache=global_cache)#, initial_var_parameters=var_parameters)
    t = time.time()

    logging.critical(result)
    print(result)
    print('Time for running: ', t - t0)

    print('Pizza')


