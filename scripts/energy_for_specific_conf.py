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

    r = 1.25
    molecule = LiH(r=r)  #frozen_els={'occupied': [0, 1], 'unoccupied': []})

    # logging
    LogUtils.log_cofig()

    df = pandas.read_csv("../results/adapt_vqe_results/vip/LiH_h_adapt_gsdqe_r=1_03-Sep-2020.csv")
    # df = pandas.read_csv("../x_sdfsd.csv")

    init_ansatz_elements = []

    # for i in range(len(df)):
    #     element = df.loc[i]['element']
    #     element_qubits = df.loc[i]['element_qubits']
    #     if element[5] == 's':
    #         init_ansatz_elements.append(SpinComplementSFExcitation(*ast.literal_eval(element_qubits), system_n_qubits=molecule.n_qubits))
    #     elif element[5] == 'd':
    #         init_ansatz_elements.append(SpinComplementDFExcitation(*ast.literal_eval(element_qubits), system_n_qubits=molecule.n_qubits))

    for i in range(len(df)):
        element = df.loc[i]['element']
        element_qubits = df.loc[i]['element_qubits']
        if element[0] == 'e' and element[4] == 's':
            init_ansatz_elements.append(EffSFExc(*ast.literal_eval(element_qubits), system_n_qubits=molecule.n_qubits))
        elif element[0] == 'e' and element[4] == 'd':
            init_ansatz_elements.append(EffDFExc(*ast.literal_eval(element_qubits), system_n_qubits=molecule.n_qubits))
        elif element[0] == 's' and element[2] == 'q':
            init_ansatz_elements.append(SQExc(*ast.literal_eval(element_qubits), system_n_qubits=molecule.n_qubits))
        elif element[0] == 'd' and element[2] == 'q':
            init_ansatz_elements.append(DQExc(*ast.literal_eval(element_qubits), system_n_qubits=molecule.n_qubits))
        else:
            print(element, element_qubits)
            raise Exception('Unrecognized ansatz element.')
    # for i in range(len(df)):
    #     excitation = QubitOperator(df.loc[i]['element'])
    #     init_ansatz_elements.append(PauliWordExcitation(excitation, system_n_qubits=molecule.n_qubits))

    ansatz_elements = init_ansatz_elements[:52]

    var_parameters = list(df['var_parameters'])[:52]
    # var_parameters = list(numpy.zeros(len(ansatz_elements)))

    optimizer = 'BFGS'
    optimizer_options = {'gtol': 1e-9}

    vqe_runner = VQERunner(molecule, backend=QiskitSim, optimizer=optimizer, optimizer_options=None,
                           print_var_parameters=False, use_ansatz_gradient=True)

    energy = vqe_runner.vqe_run(ansatz_elements=ansatz_elements, initial_var_parameters=var_parameters,
                                initial_statevector_qasm=None)

    print(energy)
