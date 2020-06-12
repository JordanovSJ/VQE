from src.vqe_runner import VQERunner
from src.molecules import H2, LiH, HF, BeH2
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
    r_0 = 1.546

    time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")
    df_data = pandas.DataFrame(columns=['r', 'E', 'fci_E', 'error', 'n_iters'])
    df_count = 0

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

    var_parameters = init_var_parameters
    energies_1 = []
    fci_energies_1 = []
    rs_1 = []
    for i in range(1, 13):
        # r = r_0 + i*0.05
        r = 2.146 + i*0.05
        molecule_params = {'distance': r}

        vqe_runner = VQERunner(molecule, backend=QiskitSimulation, molecule_geometry_params=molecule_params)
        result = vqe_runner.vqe_run(ansatz_elements, var_parameters)

        # next var parameters
        var_parameters = result.x

        fci_E = vqe_runner.fci_energy
        fci_energies_1.append(fci_E)
        E = result.fun
        energies_1.append(E)
        rs_1.append(r)

        df_data.loc[df_count] = {'r': r, 'E': E, 'fci_E': fci_E, 'error': E-fci_E, 'n_iters': result.nfev}
        df_data.to_csv('../results/dissociation_curves/{}_{}.csv'.format(molecule.name, time_stamp))
        df_count += 1

    var_parameters = init_var_parameters
    energies_2 = []
    fci_energies_2 = []
    rs_2 = []
    for i in range(0):
        r = r_0 - (1+i) * 0.025
        molecule_params = {'distance': r}

        vqe_runner = VQERunner(molecule, backend=QiskitSimulation, molecule_geometry_params=molecule_params)
        result = vqe_runner.vqe_run(ansatz_elements, var_parameters)

        # next var parameters
        var_parameters = result.x

        fci_E = result.fci_energy
        E = result.fun
        energies_2.append(E)
        fci_energies_2.append(fci_E)
        rs_2.append(r)

        df_data.loc[df_count] = {'r': r, 'E': E, 'fci_E': fci_E, 'error': E - fci_E, 'n_iters': result.nfev}
        df_data.to_csv('../results/dissociation_curves/{}_{}.csv'.format(molecule.name, time_stamp))
        df_count += 1

    energies = energies_2[::-1] + energies_1
    fci_energies = fci_energies_2[::-1] + fci_energies_1
    errors = list(numpy.array(energies) - numpy.array(fci_energies))
    rs = rs_2[::-1] + rs_1

    # df_data = pandas.DataFrame({'r': rs, 'E': energies, 'fci_E': fci_energies, 'error': errors})
    # time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")
    # df_data.to_csv('../results/{}_dis_curve_{}'.format(molecule.name, time_stamp))

    print(energies)
    print(fci_energies)
    print(rs)

    print('Bona Dea')
