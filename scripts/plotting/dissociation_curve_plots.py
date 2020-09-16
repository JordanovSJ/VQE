import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas
from src.q_systems import *

if __name__ == "__main__":

    db_data_lih_hf = pandas.read_csv('../../results/dissociation_curves/LiH_hf_03-Sep-2020.csv')
    db_data_lih_06 = pandas.read_csv('../../results/dissociation_curves/LiH_h_adapt_gsdqe_e-06_02-Sep-2020.csv')
    db_data_lih_08 = pandas.read_csv('../../results/dissociation_curves/LiH_h_adapt_gsdqe_e-08_03-Sep-2020.csv')
    db_data_lih_uccsd = pandas.read_csv('../../results/dissociation_curves/LiH_uccsd_02-Sep-2020.csv')

    # fci_Es = []
    # fci_rs = []
    # for i in range(30):
    #     r = 1 + 2.5*i/30
    #     fci_rs.append(r)
    #     fci_Es.append(LiH(r=r).fci_energy)
    #
    # plt.plot(db_data_lih_hf['r'], db_data_lih_hf['error'], label='HF', marker='*', linewidth=0.5)
    # plt.plot(db_data_lih_uccsd['r'], db_data_lih_uccsd['error'], label=r'UCCSD', marker='*', linewidth=0.5)
    # plt.plot(db_data_lih_uccsd['r'], np.zeros(11) + 468, label=r'UCCSD', marker='*', linewidth=0.5)
    plt.plot(db_data_lih_06['r'], db_data_lih_06['n_iters'], label=r'IQEB-VQE $\epsilon=10^{-6}$', marker='*', linewidth=0.5, color='green')
    plt.plot(db_data_lih_08['r'], db_data_lih_08['n_iters'], label=r'IQEB-VQE $\epsilon=10^{-8}$', marker='*', linewidth=0.5, color='red')
    # plt.plot(fci_rs, fci_Es, label='FCI energy', color='purple')
    plt.vlines([1.546], ymax=200, ymin=-100, linewidth=0.3, label='ground configuration')

    plt.xlabel(r'Li-H bond distance, $\AA$')
    # plt.ylabel(r'$E-E_{FCI}$, Hartree')
    plt.ylabel('Number of parameters')
    # plt.ylabel('Energy, Hartree')
    # plt.ylim(1e-9, 1)
    # plt.ylim(30,110)
    plt.ylim(25, 55)
    # plt.ylim(-7.90, -7.65)
    # plt.ylim(-15.6, -14.9)
    plt.xlim(0.75, 3.75)
    plt.fill_between([0.5, 3.75], 1e-9, 1e-3, color='lavender', label='chemical accuracy')
    # plt.yscale('log')
    # plt.legend(loc=9)

    plt.show()

    print('macaroni')