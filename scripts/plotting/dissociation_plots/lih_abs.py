import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas
from src.q_systems import *

if __name__ == "__main__":

    db_data_lih_hf = pandas.read_csv('../../../results/dissociation_curves/LiH_hf_03-Sep-2020.csv')
    db_data_lih_04 = pandas.read_csv('../../../results/dissociation_curves/LiH_iqeb_04.csv')
    db_data_lih_06 = pandas.read_csv('../../../results/dissociation_curves/LiH_h_adapt_gsdqe_e-06_02-Sep-2020.csv')
    db_data_lih_08 = pandas.read_csv('../../../results/dissociation_curves/LiH_h_adapt_gsdqe_e-08_03-Sep-2020.csv')
    db_data_lih_uccsd = pandas.read_csv('../../../results/dissociation_curves/LiH_uccsd_02-Sep-2020.csv')

    fci_rs = db_data_lih_06['r']
    fci_Es = db_data_lih_06['fci_E']

    # fci_Es = []
    # fci_rs = []
    # for i in range(30):
    #     r = 1 + 2.5*i/30
    #     fci_rs.append(r)
    #     fci_Es.append(LiH(r=r).fci_energy)
    #
    plt.plot(db_data_lih_hf['r'], db_data_lih_hf['E'], label='Hartree-Fock', marker='+', linewidth=0.5, color='orange')
    plt.plot(db_data_lih_uccsd['r'], db_data_lih_uccsd['E'], label=r'UCCSD', marker='+', linewidth=0.5, color='tab:brown')
    plt.plot(db_data_lih_04['r'], db_data_lih_04['E'], label=r'QEB-ADAPT-VQE $\epsilon=10^{-4}$ Har.', marker='+', linewidth=0.5, color='blue')

    plt.plot(db_data_lih_06['r'], db_data_lih_06['E'], label=r'QEB-ADAPT-VQE $\epsilon=10^{-6}$ Har.', marker='+', linewidth=0.5, color='green')
    plt.plot(db_data_lih_08['r'], db_data_lih_08['E'], label=r'QEB-ADAPT-VQE $\epsilon=10^{-8}$ Har.', marker='+', linewidth=0.5, color='red')
    plt.plot(fci_rs, fci_Es, label='FCI energy', color='purple', linewidth=0.5)
    # plt.vlines([1.546], ymax=200, ymin=-100, linewidth=0.75, color='black', label='equilibrium configuration')
    plt.fill_between([0.5, 3.75], 1e-9, 1e-3, color='lavender', label='chemical accuracy')

    # plt.xlabel(r'Li-H bond distance, $\AA$')
    plt.ylabel('Energy, Hartree', fontsize=15)
    plt.ylim(-7.90, -7.65)
    plt.xlim(0.75, 3.75)
    plt.legend(loc=2, fontsize=10.5)
    plt.grid(b=True, which='major', color='grey', linestyle='--', linewidth=0.5)

    plt.show()

    print('macaroni')