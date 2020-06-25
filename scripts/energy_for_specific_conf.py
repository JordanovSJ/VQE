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

beh2_var_pars = [3.86557742e-02,  2.40575234e-02,  2.40577222e-02,  2.89344303e-02,
                2.01598918e-02,  2.71039758e-02, -2.10362088e-02, -2.10362519e-02,
                2.71532263e-02,  1.39725253e-02, -5.08023631e-03,  3.15346769e-03,
                3.15268540e-03, -5.08068790e-03, -1.56073686e-03,  1.56017065e-03,
               -1.56062101e-03,  1.56010834e-03,  9.25733767e-04,  9.05770367e-04,
                2.22456202e-03,  7.67901879e-04,  7.67970851e-04,  2.20877250e-03,
                6.34260849e-04, -6.34861962e-04,  1.06689473e-03,  3.87889011e-04,
                1.06820789e-03, -3.88338244e-04,  2.38506681e-04,  2.38383255e-04,
               -2.27543143e-04, -2.27508539e-04,  1.82170178e-04,  8.71565328e-05,
                8.71543570e-05,  3.86302074e-02,  2.40734320e-02,  2.40732943e-02,
                2.89696322e-02,  2.01764525e-02,  2.70476403e-02, -2.10443891e-02,
               -2.10444292e-02,  2.70850551e-02,  1.39625478e-02, -5.21817800e-03,
                3.16708024e-03,  3.16937398e-03, -5.22124052e-03, -1.56684391e-03,
                1.56656281e-03, -1.56711871e-03,  1.56652500e-03,  9.40514266e-04,
                9.06030827e-04,  2.05803260e-03,  7.65976533e-04,  7.65345584e-04,
                2.04244237e-03,  6.44241939e-04, -6.44646726e-04,  1.11834289e-03,
                3.92579004e-04,  1.11744296e-03, -3.93116213e-04,  2.36599212e-04,
                2.36594268e-04, -2.22603598e-04, -2.22548668e-04,  1.97498169e-04,
                1.02656777e-04,  1.02658435e-04]


if __name__ == "__main__":

    molecule = LiH(frozen_els={'occupied': [0, 1], 'unoccupied': []})
    r = 1.546

    # logging
    LogUtils.log_cofig()

    df = pandas.read_csv("../results/adapt_vqe_results/LiH_efficient_fermi_excitation_{'occupied': [0, 1], 'unoccupied': []}_24-Jun-2020 (14:57:29.205783).csv")

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

    # ansatz_elements += ansatz_elements

    init_var_parameters = list(df['var_parameters'])
    # for i in range(len(init_var_parameters)):
    #     init_var_parameters[i] += 0.01*((init_var_parameters[i] > 0) - 0.5)*2

    init_state_qasm = QasmUtils.hf_state(molecule.n_electrons)\
                      + QasmUtils.qasm_from_ansatz_elements(init_ansatz_elements, init_var_parameters)

    ansatz_elements = [EfficientDoubleFermiExcitation([0, 1], [6, 7])]
    var_pars = [0]

    optimizer = 'L-BFGS-B'
    # optimizer = 'BFGS'
    optimizer_options = None
    optimizer_options = {'maxcor': 25, 'ftol': 1e-11, 'gtol': 1e-9, 'eps': 1e-06, 'maxfun': 1500, 'maxiter': 1500,
                         'iprint': -1, 'maxls': 25}

    vqe_runner = VQERunner(molecule, backend=QiskitSimulation, optimizer=optimizer, optimizer_options=optimizer_options,
                           print_var_parameters=True)

    energy = vqe_runner.vqe_run(ansatz_elements=ansatz_elements, initial_var_parameters=var_pars,
                                initial_statevector_qasm=init_state_qasm)

    print(energy)
