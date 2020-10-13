from src.vqe_runner import VQERunner
from src.q_systems import H2, LiH, HF, BeH2
from src.ansatz_element_lists import *
from src.backends import QiskitSim
from src.utils import LogUtils
from src.iter_vqe_utils import *

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

    r = 1.546
    molecule = LiH(r=r)  #frozen_els={'occupied': [0, 1], 'unoccupied': []})

    # logging
    LogUtils.log_config()

    df = pandas.read_csv("../results/iter_vqe_results/vip/LiH_g_adapt_gsdqe_comp_exc_19-Sep-2020.csv")
    # df = pandas.read_csv("../x_sdfsd.csv")

    ansatz, var_parameters = IterVQEDataUtils.get_ansatz_from_data_frame(df, molecule)

    ansatz = ansatz[:15]
    # var_parameters = list(df['var_parameters'])[:49]
    var_parameters = var_parameters[:15]

    optimizer = 'BFGS'
    optimizer_options = {'gtol': 1e-8}

    vqe_runner = VQERunner(molecule, backend=QiskitSim, optimizer=optimizer, optimizer_options=None,
                           print_var_parameters=False, use_ansatz_gradient=True)

    energy = vqe_runner.vqe_run(ansatz=ansatz, initial_var_parameters=var_parameters,
                                init_state_qasm=None)

    print(energy)
