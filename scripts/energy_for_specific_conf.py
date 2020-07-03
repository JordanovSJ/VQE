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

beh2_var_pars =[ 0.15165819,  0.09736567,  0.10228814,  0.00339765,  0.01015889,
        0.22078285, -0.01306282, -0.00261087,  0.22717355, -0.04463201,
       -0.21654193, -0.0035964 ,  0.00554479, -0.21853316,  0.03356841,
       -0.02597601,  0.0263118 , -0.02846377,  0.00750505,  0.01461668,
        0.02029264,  0.00494764,  0.00589723, -0.00233985,  0.00034058,
        0.00181185,  0.00973092,  0.00516921,  0.0322556 , -0.00172135,
       -0.0006497 ,  0.00279393,  0.00882293,  0.00922108,  0.01697921,
        0.02370743,  0.01712358, -0.07632847, -0.05274612, -0.05821446,
        0.06638091,  0.03089694, -0.17389571, -0.03158068, -0.04257354,
       -0.17068236,  0.07219871,  0.18029454,  0.0100354 ,  0.00167378,
        0.18389292, -0.03479518,  0.02746876, -0.02798771,  0.02963268,
       -0.00515208, -0.01265535, -0.00584082, -0.00323024, -0.0043157 ,
        0.01916254,  0.00087151, -0.00310928, -0.00970359, -0.00358728,
       -0.03133639,  0.00028345,  0.00139419, -0.00216603, -0.00903253,
       -0.00935104, -0.01840185, -0.02189118, -0.02022312]


if __name__ == "__main__":

    molecule = LiH()  #frozen_els={'occupied': [0, 1], 'unoccupied': []})
    # r = 1.546

    # logging
    LogUtils.log_cofig()

    df = pandas.read_csv("../results/adapt_vqe_results/LiH_grad_adapt_pwe_27-Jun-2020_updated_2.csv")

    init_ansatz_elements = []
    # for i in range(len(df)):
    #     element = df.loc[i]['element']
    #     element_qubits = df.loc[i]['element_qubits']
    #     if element[0] == 'e' and element[4] == 's':
    #         init_ansatz_elements.append(EfficientSingleFermiExcitation(*ast.literal_eval(element_qubits)))
    #     elif element[0] == 'e' and element[4] == 'd':
    #         init_ansatz_elements.append(EfficientDoubleFermiExcitation(*ast.literal_eval(element_qubits)))
    #     elif element[0] == 's' and element[2] == 'q':
    #         init_ansatz_elements.append(SingleQubitExcitation(*ast.literal_eval(element_qubits)))
    #     elif element[0] == 'd' and element[2] == 'q':
    #         init_ansatz_elements.append(DoubleQubitExcitation(*ast.literal_eval(element_qubits)))
    #     else:
    #         print(element, element_qubits)
    #         raise Exception('Unrecognized ansatz element.')
    for i in range(len(df)):
        excitation = QubitOperator(df.loc[i]['element'])
        init_ansatz_elements.append(PauliWordExcitation(excitation))

    init_ansatz_elements = init_ansatz_elements
    # init_ansatz_elements += init_ansatz_elements
    # ansatz_elements += ansatz_elements

    init_var_parameters = list(df['var_parameters'])
    # init_var_parameters += list(numpy.zeros(len(init_var_parameters)))
    # for i in range(len(init_var_parameters)):
    #     init_var_parameters[i] += 0.01*((init_var_parameters[i] > 0) - 0.5)*2

    # init_state_qasm = QasmUtils.hf_state(molecule.n_electrons)\
    #                   + QasmUtils.qasm_from_ansatz_elements(init_ansatz_elements, init_var_parameters)

    # ansatz_elements = [EfficientDoubleFermiExcitation([0, 1], [6, 7])]
    # var_pars = [0]

    optimizer = 'BFGS'
    # optimizer = 'BFGS'
    optimizer_options = {'gtol': 1e-6}
    # optimizer_options = {'maxcor': 25, 'ftol': 1e-11, 'gtol': 1e-9, 'eps': 1e-06, 'maxfun': 1500, 'maxiter': 1500,
    #                      'iprint': -1, 'maxls': 25}

    vqe_runner = VQERunner(molecule, backend=QiskitSim, optimizer=optimizer, optimizer_options=optimizer_options,
                           print_var_parameters=True)

    energy = vqe_runner.vqe_run(ansatz_elements=init_ansatz_elements, initial_var_parameters=init_var_parameters,
                                initial_statevector_qasm=None)

    print(energy)
