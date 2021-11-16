import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas
from src.q_systems import *
from src.molecules.molecules import *

if __name__ == "__main__":

    db_data_lih_qeuccsd = pandas.read_csv('../../../results/dissociation_curves/H6_quccsd_big_03-Jun-2021.csv')
    db_data_lih_uccsd = pandas.read_csv('../../../results/dissociation_curves/H6_uccsd_big_03-Jun-2021.csv')

    plt.plot(db_data_lih_qeuccsd['r'], db_data_lih_qeuccsd['E'], label=r'QUCC', marker='*', linewidth=0.5, color='blue')
    plt.plot(db_data_lih_uccsd['r'], db_data_lih_uccsd['E'], label=r'UCC', marker='+', linewidth=0.5, color='red')
    # plt.plot(fci_rs, fci_Es, label='FCI energy', color='purple')
    # plt.vlines([1.546], ymax=200, ymin=-100, linewidth=0.75, color='black', label='ground configuration')
    plt.fill_between([0.5, 3.75], 1e-9, 1e-3, color='lavender', label='chemical accuracy')

    # plt.xlabel(r'Li-H bond distance, $\AA$')
    # plt.ylabel('Energy, Hartree')
    plt.ylim(-3.3, -2.2)
    plt.xlim(0.25, 3.25)
    plt.grid(b=True, which='major', color='grey', linestyle='--', linewidth=0.5)

    # plt.legend()
    plt.show()

    print('macaroni')