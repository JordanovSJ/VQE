from src.vqe_runner import VQERunner
from src.q_systems import H2, LiH, HF, BeH2
from src.ansatz_element_lists import *
from src.backends import QiskitSim
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

    time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")
    df_data = pandas.DataFrame(columns=['r', 'E', 'fci_E', 'error', 'n_iters'])

    # logging
    LogUtils.log_config()

    df = pandas.read_csv('../results/iter_vqe_results/vip/BeH2_h_adapt_gsdqe_13-Aug-2020.csv')
    #
    ansatz = []
    for i in range(len(df)):
        element = df.loc[i]['element']
        element_qubits = df.loc[i]['element_qubits']
        if element[0] == 'e' and element[4] == 's':
            ansatz.append(EffSFExc(*ast.literal_eval(element_qubits), system_n_qubits=molecule.n_qubits))
        elif element[0] == 'e' and element[4] == 'd':
            ansatz.append(EffDFExc(*ast.literal_eval(element_qubits), system_n_qubits=molecule.n_qubits))
        elif element[0] == 's' and element[2] == 'q':
            ansatz.append(SQExc(*ast.literal_eval(element_qubits), system_n_qubits=molecule.n_qubits))
        elif element[0] == 'd' and element[2] == 'q':
            ansatz.append(DQExc(*ast.literal_eval(element_qubits), system_n_qubits=molecule.n_qubits))
        else:
            print(element, element_qubits)
            raise Exception('Unrecognized ansatz element.')

    # ansatz = ansatz[:74]  # 74 for 1e-8
    # ansatz = UCCSD(molecule.n_qubits, molecule.n_electrons).get_ansatz_elements()
    ansatz = []
    var_parameters = list(numpy.zeros(len(ansatz)))

    optimizer = 'BFGS'
    optimizer_options = {'gtol': 10e-8}

    energies = []
    fci_energies = []
    rs = [0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5]
    df_count = 0

    for r in rs:
        molecule = BeH2(r=r)

        vqe_runner = VQERunner(molecule, backend=QiskitSim, use_ansatz_gradient=True, optimizer=optimizer,
                               optimizer_options=optimizer_options)

        result = vqe_runner.vqe_run(ansatz, var_parameters)

        fci_E = molecule.fci_energy
        fci_energies.append(fci_E)

        # next var parameters
        if len(ansatz) == 0:
            var_parameters = []
            E = result
            n_iters = 1
        else:
            var_parameters = list(result.x)
            E = result.fun
            n_iters = result.nfev

        energies.append(E)

        df_data.loc[df_count] = {'r': r, 'E': E, 'fci_E': fci_E, 'error': E-fci_E, 'n_iters': n_iters}
        df_data.to_csv('../results/dissociation_curves/{}_{}.csv'.format(molecule.name, time_stamp))
        df_count += 1

    # df_data = pandas.DataFrame({'r': rs, 'E': energies, 'fci_E': fci_energies, 'error': errors})
    # time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")
    # df_data.to_csv('../results/{}_dis_curve_{}'.format(molecule.name, time_stamp))

    print('Bona Dea')
