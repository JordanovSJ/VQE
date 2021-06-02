from src.vqe_runner import VQERunner
from src.molecules.molecules import *
from src.ansatz_element_sets import *
from src.backends import QiskitSimBackend, MatrixCacheBackend
from src.utils import LogUtils
from src.cache import GlobalCache

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

    molecule = H6()

    time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")
    df_data = pandas.DataFrame(columns=['r', 'E', 'fci_E', 'error', 'n_iters'])

    # logging
    LogUtils.log_config()

    # ansatz = ansatz[:74]  # 74 for 1e-8
    ansatz = UCCSDExcitations(molecule.n_qubits, molecule.n_electrons, ansatz_element_type='eff_f_exc').get_excitations()
    var_parameters = list(numpy.zeros(len(ansatz)))

    optimizer = 'BFGS'
    optimizer_options = {'gtol': 10e-8}

    energies = []
    fci_energies = []
    rs = [0.5, 0.75] # 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
    df_count = 0

    for r in rs:
        molecule = H6(r=r)

        vqe_runner = VQERunner(molecule, backend=MatrixCacheBackend, use_ansatz_gradient=True, optimizer=optimizer,
                               optimizer_options=optimizer_options)
        global_cache = GlobalCache(molecule)
        global_cache.calculate_exc_gen_sparse_matrices_dict(ansatz)

        result = vqe_runner.vqe_run(ansatz, var_parameters, cache=global_cache)

        del global_cache

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
