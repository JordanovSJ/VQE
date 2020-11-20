import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas
from src.q_systems import *

if __name__ == "__main__":

    db_data_lih_hf = pandas.read_csv('../../../results/dissociation_curves/LiH_hf_03-Sep-2020.csv')
    db_data_lih_06 = pandas.read_csv('../../../results/dissociation_curves/LiH_h_adapt_gsdqe_e-06_02-Sep-2020.csv')
    db_data_lih_08 = pandas.read_csv('../../../results/dissociation_curves/LiH_h_adapt_gsdqe_e-08_03-Sep-2020.csv')
    db_data_lih_uccsd = pandas.read_csv('../../../results/dissociation_curves/LiH_uccsd_02-Sep-2020.csv')

    # fci_Es = []
    # fci_rs = []
    # for i in range(30):
    #     r = 1 + 2.5*i/30
    #     fci_rs.append(r)
    #     fci_Es.append(LiH(r=r).fci_energy)
    #
    plt.plot(db_data_lih_hf['r'], db_data_lih_hf['error'], label='HF', marker='+', linewidth=0.5)
    plt.plot(db_data_lih_uccsd['r'], db_data_lih_uccsd['error'], label=r'UCCSD', marker='+', linewidth=0.5)
    plt.plot(db_data_lih_06['r'], db_data_lih_06['error'], label=r'IQEB-VQE $\epsilon=10^{-6}$', marker='+', linewidth=0.5, color='green')
    plt.plot(db_data_lih_08['r'], db_data_lih_08['error'], label=r'IQEB-VQE $\epsilon=10^{-8}$', marker='+', linewidth=0.5, color='red')

    plt.vlines([1.546], ymax=200, ymin=-100, linewidth=0.75, color='black', label='ground configuration')
    plt.fill_between([0.5, 3.75], 1e-9, 1e-3, color='lavender', label='chemical accuracy')

    plt.xlabel(r'Li-H bond distance, $\AA$')
    plt.ylabel(r'$E-E_{FCI}$, Hartree')
    plt.ylim(1e-9, 1)
    plt.xlim(0.75, 3.75)
    plt.yscale('log')
    plt.grid(b=True, which='major', color='grey', linestyle='--', linewidth=0.5)

    plt.show()

    print('macaroni')