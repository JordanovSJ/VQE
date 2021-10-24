from src.vqe_runner import VQERunner
from src.molecules.molecules import H2, LiH, HF, BeH2
from src.ansatz_element_sets import *
from src.backends import QiskitSimBackend, MatrixCacheBackend
from src.utils import LogUtils
from src.iter_vqe_utils import DataUtils
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

    molecule = LiH()
    excited_state = 1
    time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")
    df_data = pandas.DataFrame(columns=['r', 'E', 'fci_E', 'error', 'n_iters'])

    # logging
    LogUtils.log_config()

    # ansatz = ansatz[:74]  # 74 for 1e-8
    ansatz = UCCSDExcitations(molecule.n_qubits, molecule.n_electrons, ansatz_element_type='eff_f_exc').get_all_elements()
    var_parameters = list(numpy.zeros(len(ansatz)))

    optimizer = 'BFGS'
    optimizer_options = {'gtol': 10e-8}

    energies = []
    exact_energies = []
    rs = [1, 1.25, 1.546, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5]
    gs_dfs = []
    gs_dfs.append(pandas.read_csv('../../results/iter_vqe_results/LiH_iqeb_eff_f_exc_r=1_15-Mar-2021.csv'))
    gs_dfs.append(pandas.read_csv('../../results/iter_vqe_results/LiH_iqeb_q_exc_r=1.25_19-Nov-2020.csv'))
    gs_dfs.append(pandas.read_csv('../../results/iter_vqe_results/LiH_iqeb_eff_f_exc_r=1546_15-Mar-2021.csv'))
    gs_dfs.append(pandas.read_csv('../../results/iter_vqe_results/LiH_iqeb_q_exc_r=1.75_19-Nov-2020.csv'))
    gs_dfs.append(pandas.read_csv('../../results/iter_vqe_results/LiH_adapt_f_exc_r=2_01-Apr-2021.csv'))
    gs_dfs.append(pandas.read_csv('../../results/iter_vqe_results/LiH_iqeb_q_exc_r=2.25_19-Nov-2020.csv'))
    gs_dfs.append(pandas.read_csv('../../results/iter_vqe_results/LiH_iqeb_q_exc_r=2.5_19-Nov-2020.csv'))
    gs_dfs.append(pandas.read_csv('../../results/iter_vqe_results/LiH_iqeb_q_exc_r=2.75_19-Nov-2020.csv'))
    gs_dfs.append(pandas.read_csv('../../results/iter_vqe_results/LiH_iqeb_q_exc_no_comps_n=1_r=3_18-Mar-2021.csv'))
    gs_dfs.append(pandas.read_csv('../../results/iter_vqe_results/LiH_iqeb_q_exc_r=3.25_19-Nov-2020.csv'))
    gs_dfs.append(pandas.read_csv('../../results/iter_vqe_results/LiH_iqeb_q_exc_r=3.5_19-Nov-2020.csv'))

    # rs = [0.75, 1, 1.316, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
    # gs_dfs = []
    # gs_dfs.append(pandas.read_csv('../results/iter_vqe_results/vip/BeH2_h_adapt_gsdqe_comp_pair_r=075_09-Oct-2020.csv'))
    # gs_dfs.append(pandas.read_csv('../results/iter_vqe_results/vip/BeH2_h_adapt_gsdqe_r=1_04-Sep-2020.csv'))
    # gs_dfs.append(pandas.read_csv('../results/iter_vqe_results/BeH2_iqeb_q_exc_n=1_r=1316_17-Mar-2021.csv'))
    # gs_dfs.append(pandas.read_csv('../results/iter_vqe_results/BeH2_iqeb_vqe_r=15_19-Nov-2020.csv'))
    # gs_dfs.append(pandas.read_csv('../results/iter_vqe_results/BeH2_iqeb_vqe_r=175_19-Nov-2020.csv'))
    # gs_dfs.append(pandas.read_csv('../results/iter_vqe_results/BeH2_iqeb_vqe_r=2_19-Nov-2020.csv'))
    # gs_dfs.append(pandas.read_csv('../results/iter_vqe_results/BeH2_iqeb_vqe_r=225_20-Nov-2020.csv'))
    # gs_dfs.append(pandas.read_csv('../results/iter_vqe_results/BeH2_iqeb_vqe_r=25_20-Nov-2020.csv'))
    # gs_dfs.append(pandas.read_csv('../results/iter_vqe_results/BeH2_iqeb_vqe_r=275_20-Nov-2020.csv'))
    # gs_dfs.append(pandas.read_csv('../results/iter_vqe_results/BeH2_iqeb_eff_f_exc_n=1_r=3_22-Mar-2021.csv'))

    df_count = 0

    for j, r in enumerate(rs):
        print('r= ', r)
        molecule = LiH(r=r)
        ground_state = DataUtils.ansatz_from_data_frame(gs_dfs[j], molecule)
        molecule.H_lower_state_terms = [[abs(molecule.hf_energy) * 2, ground_state]]

        vqe_runner = VQERunner(molecule, backend=MatrixCacheBackend, use_ansatz_gradient=True, optimizer=optimizer,
                               optimizer_options=optimizer_options)
        global_cache = GlobalCache(molecule, excited_state=excited_state)
        global_cache.calculate_exc_gen_sparse_matrices_dict(ansatz)

        result = vqe_runner.vqe_run(ansatz, var_parameters, cache=global_cache, excited_state=excited_state)

        del global_cache

        exact_E = molecule.calculate_energy_eigenvalues(excited_state + 1)[excited_state]
        exact_energies.append(exact_E)

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

        df_data.loc[df_count] = {'r': r, 'E': E, 'fci_E': exact_E, 'error': E - exact_E, 'n_iters': n_iters}
        df_data.to_csv('../results/dissociation_curves/{}_exc_{}_uccsd_{}.csv'.format(molecule.name,excited_state, time_stamp))
        df_count += 1

    # df_data = pandas.DataFrame({'r': rs, 'E': energies, 'fci_E': fci_energies, 'error': errors})
    # time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")
    # df_data.to_csv('../results/{}_dis_curve_{}'.format(molecule.name, time_stamp))

    print('Bona Dea')
