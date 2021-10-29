import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas
from src.molecular_system import *
from src.molecules.molecules import *

if __name__ == "__main__":

    db_data_beh2_qeuccsd = pandas.read_csv('../../../results/dissociation_curves/BeH2_qeuccsd_08-Mar-2021.csv')
    db_data_beh2_uccsd = pandas.read_csv('../../../results/dissociation_curves/BeH2_uccsd_08-Sep-2020.csv')

    fci_Es = []
    fci_rs = []
    # for i in range(30):
    #     r = 1 + 2.5*i/30
    #     fci_rs.append(r)
    #     fci_Es.append(BeH2(r=r).fci_energy)
    #
    plt.plot(db_data_beh2_qeuccsd['r'], db_data_beh2_qeuccsd['E'], label=r'QUCC', marker='*', linewidth=0.5, color='blue')
    plt.plot(db_data_beh2_uccsd['r'], db_data_beh2_uccsd['E'], label=r'UCC', marker='+', linewidth=0.5, color='red')
    # plt.plot(fci_rs, fci_Es, label='FCI energy', color='purple')
    # plt.vlines([1.316], ymax=200, ymin=-100, linewidth=0.75, color='black', label='ground configuration')
    plt.fill_between([0.5, 3.75], 1e-9, 1e-3, color='lavender', label='chemical accuracy')

    # plt.xlabel(r'Be-H bond distance, $\AA$')
    # plt.ylabel('Energy, Hartree')
    plt.ylim(-15.6, -15)
    plt.xlim(0.5, 3.5)
    plt.grid(b=True, which='major', color='grey', linestyle='--', linewidth=0.5)

    plt.show()

    print('macaroni')