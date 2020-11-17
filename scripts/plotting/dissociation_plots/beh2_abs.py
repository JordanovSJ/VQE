import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas
from src.q_systems import *

if __name__ == "__main__":

    db_data_lih_hf = pandas.read_csv('../../../results/dissociation_curves/BeH2_hf_06-Oct-2020.csv')
    db_data_lih_06 = pandas.read_csv('../../../results/dissociation_curves/BeH2_h_adapt_gsdqe_e-06_03-Sep-2020.csv')
    db_data_lih_08 = pandas.read_csv('../../../results/dissociation_curves/BeH2_h_adapt_gsdqe_e-08_15-Sep-2020.csv')
    db_data_lih_uccsd = pandas.read_csv('../../../results/dissociation_curves/BeH2_uccsd_08-Sep-2020.csv')

    fci_Es = []
    fci_rs = []
    for i in range(30):
        r = 1 + 2.5*i/30
        fci_rs.append(r)
        fci_Es.append(BeH2(r=r).fci_energy)
    #
    plt.plot(db_data_lih_hf['r'], db_data_lih_hf['E'], label='HF', marker='+', linewidth=0.5)
    plt.plot(db_data_lih_uccsd['r'], db_data_lih_uccsd['E'], label=r'UCCSD', marker='+', linewidth=0.5)
    plt.plot(db_data_lih_06['r'], db_data_lih_06['E'], label=r'IQEB-VQE $\epsilon=10^{-6}$', marker='+', linewidth=0.5, color='green')
    plt.plot(db_data_lih_08['r'], db_data_lih_08['E'], label=r'IQEB-VQE $\epsilon=10^{-8}$', marker='+', linewidth=0.5, color='red')
    plt.plot(fci_rs, fci_Es, label='FCI energy', color='purple', linewidth=0.3)
    plt.vlines([1.316], ymax=200, ymin=-100, linewidth=0.75, color='black', label='ground configuration')
    plt.fill_between([0.5, 3.75], 1e-9, 1e-3, color='lavender', label='chemical accuracy')

    plt.xlabel(r'Be-H bond distance, $\AA$')
    plt.ylabel('Energy, Hartree')
    plt.ylim(-15.6, -14.9)
    plt.xlim(0.5, 3.5)
    plt.grid(b=True, which='major', color='grey', linestyle='--', linewidth=0.5)

    plt.show()

    print('macaroni')