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


if __name__ == "__main__":

    molecule = HF

    # logging
    LogUtils.log_cofig()

    ansatz_elements = [DoubleQubitExcitation([4, 5], [10, 11]), DoubleQubitExcitation([3, 4], [10, 11]),
                       DoubleQubitExcitation([2, 3], [10, 11]), DoubleQubitExcitation([2, 5], [10, 11]),
                       DoubleQubitExcitation([6, 7], [10, 11]), DoubleQubitExcitation([8, 9], [10, 11]),
                       SingleQubitExcitation(5, 11), SingleQubitExcitation(4, 10), SingleQubitExcitation(3, 11),
                       DoubleQubitExcitation([0, 1], [10, 11]), DoubleQubitExcitation([0, 3], [10, 11]),
                       DoubleQubitExcitation([1, 2], [10, 11]),
                       SingleQubitExcitation(2, 10),
                       DoubleQubitExcitation([1, 4], [10, 11]), DoubleQubitExcitation([0, 5], [10, 11])]

    initial_var_parameters = [-0.1348821853363791, -0.030308892233037205, -0.02438520854419213, 0.030421484789544904,
                              -0.017923593172760655, -0.017926644529627764, -0.016163632894877294, 0.015997841462461835,
                              -0.0031624799369282075, -0.0004065477078839333, 0.0004266015854361852, -0.0005520783699704946,
                              0.001601246718874758, -9.07917533487873e-05, 8.196598228859834e-06]

    var_parameters = initial_var_parameters
    energies_1 = []
    fci_energies_1 = []
    rs_1 = []
    for i in range(10):
        r = 0.995 + i*0.025
        molecule_params = {'distance': r}

        vqe_runner = VQERunner(molecule, backend=QiskitSimulation, molecule_geometry_params=molecule_params)
        result = vqe_runner.vqe_run(ansatz_elements, var_parameters)

        # next var parameters
        var_parameters = result.x

        fci_energies_1.append(vqe_runner.fci_energy)
        energies_1.append(result.fun)
        rs_1.append(r)

    var_parameters = initial_var_parameters
    energies_2 = []
    fci_energies_2 = []
    rs_2 = []
    for i in range(10):
        r = 0.995 - (1+i) * 0.025
        molecule_params = {'distance': r}

        vqe_runner = VQERunner(molecule, backend=QiskitSimulation, molecule_geometry_params=molecule_params)
        result = vqe_runner.vqe_run(ansatz_elements, var_parameters)

        # next var parameters
        var_parameters = result.x

        energies_2.append(result.fun)
        fci_energies_2.append(vqe_runner.fci_energy)
        rs_2.append(r)

    energies = energies_2[::-1] + energies_1
    fci_energies = fci_energies_2[::-1] + fci_energies_1
    errors = list(numpy.array(energies) - numpy.array(fci_energies))
    rs = rs_2[::-1] + rs_1

    df_data = pandas.DataFrame({'r': rs, 'E': energies, 'fci_E': fci_energies, 'error': errors})
    time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")
    df_data.to_csv('../results/{}_dis_curve_{}'.format(molecule.name, time_stamp))

    print(energies)
    print(fci_energies)
    print(rs)

    print('Bona Dea')
