from src.vqe_runner import VQERunner
from src.q_systems import H2, LiH, HF, BeH2
from src.ansatz_element_lists import UCCGSD, UCCSD, DoubleExchange, SQExc, DQExc
from src.backends import QiskitSim
from src.utils import LogUtils
from src.ansatz_element_lists import *

import ast
import matplotlib.pyplot as plt

import logging
import time
import numpy
import pandas
import datetime
import scipy
import qiskit
from functools import partial


if __name__ == "__main__":

    molecule = LiH()

    df = pandas.read_csv("../results/adapt_vqe_results/vip/LiH_h_adapt_gsdqe_27-Jul-2020.csv")

    init_ansatz_elements = []
    for i in range(len(df)):
        element = df.loc[i]['element']
        element_qubits = df.loc[i]['element_qubits']
        if element[0] == 'e' and element[4] == 's':
            init_ansatz_elements.append(EffSFExc(*ast.literal_eval(element_qubits)))
        elif element[0] == 'e' and element[4] == 'd':
            init_ansatz_elements.append(EffDFExc(*ast.literal_eval(element_qubits)))
        elif element[0] == 's' and element[2] == 'q':
            init_ansatz_elements.append(SQExc(*ast.literal_eval(element_qubits)))
        elif element[0] == 'd' and element[2] == 'q':
            init_ansatz_elements.append(DQExc(*ast.literal_eval(element_qubits)))
        else:
            print(element, element_qubits)
            raise Exception('Unrecognized ansatz element.')

    ansatz_elements = init_ansatz_elements[:1]

    optimizer = 'BFGS'
    optimizer_options = {'gtol': 1e-8}

    vqe_runner = VQERunner(molecule, backend=QiskitSim, optimizer=optimizer, optimizer_options=None,
                           print_var_parameters=True, use_ansatz_gradient=True)

    angles = (numpy.arange(40) - 20)*numpy.pi/200 - 0.1
    energies = []
    for angle in angles:
        energies.append(vqe_runner.get_energy([angle],ansatz_elements))

    plt.plot(angles, energies)
    plt.xlabel(r'Angle, [$\pi$]')
    plt.ylabel('Energy, [Hartree]')
    plt.title('Mol.: {}, ansatz_element: {}: {}'.format(molecule.name, len(ansatz_elements), ansatz_elements[-1].element))
    plt.show()
