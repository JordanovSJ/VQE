from src.vqe_runner import VQERunner
from src.q_systems import H2, LiH, HF, BeH2
from src.ansatz_element_lists import *
from src.backends import QiskitSim
from src.utils import LogUtils

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

    molecule = BeH2()  #frozen_els={'occupied': [0, 1], 'unoccupied': []})
    # r = 1.546

    # logging
    LogUtils.log_cofig()

    df = pandas.read_csv("../results/adapt_vqe_results/BeH2_h_qe_26-Jul-2020.csv")

    init_ansatz_elements = []
    for i in range(len(df)):
        element = df.loc[i]['element']
        element_qubits = df.loc[i]['element_qubits']
        if element[0] == 'e' and element[4] == 's':
            init_ansatz_elements.append(EfficientSingleFermiExcitation(*ast.literal_eval(element_qubits)))
        elif element[0] == 'e' and element[4] == 'd':
            init_ansatz_elements.append(EfficientDoubleFermiExcitation(*ast.literal_eval(element_qubits)))
        elif element[0] == 's' and element[2] == 'q':
            init_ansatz_elements.append(SingleQubitExcitation(*ast.literal_eval(element_qubits)))
        elif element[0] == 'd' and element[2] == 'q':
            init_ansatz_elements.append(DoubleQubitExcitation(*ast.literal_eval(element_qubits)))
        else:
            print(element, element_qubits)
            raise Exception('Unrecognized ansatz element.')
    # for i in range(len(df)):
    #     excitation = QubitOperator(df.loc[i]['element'])
    #     init_ansatz_elements.append(PauliWordExcitation(excitation))

    init_ansatz_elements = init_ansatz_elements

    init_var_parameters = list(df['var_parameters'])

    ansatz_elements = []
    var_parameters = []
    for i in range(len(init_var_parameters)):
        if abs(init_var_parameters[i]) > 1e-4:
            ansatz_elements.append(init_ansatz_elements[i])
            var_parameters.append(init_var_parameters[i])

    optimizer = 'BFGS'
    # optimizer = 'BFGS'
    optimizer_options = {'gtol': 1e-9}
    # optimizer_options = {'maxcor': 25, 'ftol': 1e-11, 'gtol': 1e-9, 'eps': 1e-06, 'maxfun': 1500, 'maxiter': 1500,
    #                      'iprint': -1, 'maxls': 25}

    vqe_runner = VQERunner(molecule, backend=QiskitSim, optimizer=optimizer, optimizer_options=optimizer_options,
                           print_var_parameters=False)

    energy = vqe_runner.vqe_run(ansatz_elements=ansatz_elements, initial_var_parameters=var_parameters,
                                initial_statevector_qasm=None)

    print(energy)
