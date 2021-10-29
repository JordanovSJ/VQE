import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas
from src.molecular_system import *
from src.molecules.molecules import *

if __name__ == "__main__":

    db_data_lih_qeuccsd = pandas.read_csv('../../../results/dissociation_curves/LiH_qeuccsd_08-Mar-2021.csv')
    db_data_lih_uccsd = pandas.read_csv('../../../results/dissociation_curves/LiH_uccsd_02-Sep-2020.csv')

    plt.plot(db_data_lih_qeuccsd['r'], db_data_lih_qeuccsd['E'], label=r'QUCC', marker='*', linewidth=0.5, color='blue')
    plt.plot(db_data_lih_uccsd['r'], db_data_lih_uccsd['E'], label=r'UCC', marker='+', linewidth=0.5, color='red')
    # plt.plot(fci_rs, fci_Es, label='FCI energy', color='purple')
    # plt.vlines([1.546], ymax=200, ymin=-100, linewidth=0.75, color='black', label='ground configuration')
    plt.fill_between([0.5, 3.75], 1e-9, 1e-3, color='lavender', label='chemical accuracy')

    # plt.xlabel(r'Li-H bond distance, $\AA$', fontsize=15)
    plt.ylabel('Energy, Hartree', fontsize=15)
    plt.ylim(-7.89, -7.73)
    plt.xlim(0.75, 3.75)
    plt.grid(b=True, which='major', color='grey', linestyle='--', linewidth=0.5)

    plt.legend(fontsize=15)
    plt.show()

    print('macaroni')