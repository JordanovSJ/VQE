from src.vqe_runner import VQERunner
from src.q_systems import H2, LiH, HF, BeH2
from src.ansatz_element_sets import *
from src.backends import QiskitSim
from src.utils import LogUtils
from src.iter_vqe_utils import *
from src.cache import *

import matplotlib.pyplot as plt

from openfermion import QubitOperator
import logging
import time
import numpy
import pandas
import datetime
import scipy
import qiskit
from functools import partial
import ast


if __name__ == "__main__":

    r = 3
    molecule = LiH(r=r)  #frozen_els={'occupied': [0, 1], 'unoccupied': []})

    # logging
    LogUtils.log_config()

    df = pandas.read_csv("../results/iter_vqe_results/vip/LiH_g_adapt_gsdfe_comp_exc_r=3_30-Oct-2020.csv")
    # df = pandas.read_csv("../results/iter_vqe_results/vip/LiH_g_adapt_gsdfe_comp_exc_16-Sep-2020.csv")
    # df = pandas.read_csv("../results/iter_vqe_results/vip/LiH_h_adapt_gsdqe_comp_pairs_r=1_24-Sep-2020.csv")

    state = DataUtils.ansatz_from_data_frame(df, molecule)
    ansatz = state.elements
    var_parameters = state.parameters
    ansatz = ansatz

    # var_parameters = list(df['var_parameters'])[:49]
    var_parameters = var_parameters

    global_cache = GlobalCache(molecule)
    global_cache.calculate_exc_gen_sparse_matrices_dict(ansatz)
    global_cache.calculate_commutators_sparse_matrices_dict(ansatz)

    optimizer = 'BFGS'
    optimizer_options = {'gtol': 1e-8}

    vqe_runner = VQERunner(molecule, optimizer=optimizer, optimizer_options=None,
                           print_var_parameters=False, use_ansatz_gradient=True)

    energy = vqe_runner.vqe_run(ansatz=ansatz, init_guess_parameters=var_parameters,
                                init_state_qasm=None, cache=global_cache)

    print(energy)
