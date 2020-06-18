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

    molecule = BeH2()
    r = 1.546

    # logging
    LogUtils.log_cofig()

    df = pandas.read_csv('../results/adapt_vqe_results/BeH2_energy_adapt_SDQE_10-Jun-2020.csv')

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

    optimizer = 'Nelder-Mead'
    optimizer_options = {'adaptive': True}

    vqe_runner = VQERunner(molecule, backend=QiskitSimulation, optimizer=optimizer, optimizer_options=optimizer_options)

    energy = vqe_runner.vqe_run(ansatz_elements=ansatz_elements, initial_var_parameters=init_var_parameters)

    print(energy)
