from src.vqe_runner import VQERunner
from src.q_systems import H2, LiH, HF, BeH2
from src.ansatz_element_lists import *
from src.backends import QiskitSimulation
from src.utils import LogUtils

import matplotlib.pyplot as plt

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

    molecule = LiH
    r = 1.546

    # logging
    LogUtils.log_cofig()

    df = pandas.read_csv('../results/adapt_vqe_results/LiH_SDEFE_05-Jun-2020.csv')

    ansatz_elements = []
    for i in range(len(df)):
        element = df.loc[i]['element']
        element_qubits = df.loc[i]['element_qubits']
        if element[0] == 'e' and element[4] == 's':
            ansatz_elements.append(EfficientSingleFermiExcitation(*ast.literal_eval(element_qubits)))
        elif element[0] == 'e' and element[4] == 'd':
            ansatz_elements.append(EfficientDoubleFermiExcitation(*ast.literal_eval(element_qubits)))
        elif element[0] == 's' and element[2] == 'q':
            ansatz_elements.append(SingleQubitExcitation(*ast.literal_eval(element_qubits)))
        elif element[0] == 'd' and element[2] == 'q':
            ansatz_elements.append(DoubleQubitExcitation(*ast.literal_eval(element_qubits)))
        else:
            print(element, element_qubits)
            raise Exception('Unrecognized ansatz element.')

    init_var_parameters = list(df['var_parameters'])

    optimizer = 'L-BFGS-B'
    optimizer_options = {'maxcor': 20, 'ftol': 1e-10, 'gtol': 1e-08, 'eps': 1e-03, 'maxfun': 1500, 'maxiter': 1000,
                         'iprint': -1, 'maxls': 10}

    vqe_runner = VQERunner(molecule, backend=QiskitSimulation, molecule_geometry_params={'distance': r}, optimizer=optimizer,
                           optimizer_options=optimizer_options)

    energy = vqe_runner.vqe_run(ansatz_elements=ansatz_elements)#, initial_var_parameters=init_var_parameters)

    print(energy)
