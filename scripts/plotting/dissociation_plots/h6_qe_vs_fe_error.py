import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas
from src.q_system import *

if __name__ == "__main__":

    db_data_lih_qeuccsd = pandas.read_csv('../../../results/dissociation_curves/H6_quccsd_big_03-Jun-2021.csv')
    db_data_lih_uccsd = pandas.read_csv('../../../results/dissociation_curves/H6_uccsd_big_2_03-Jun-2021.csv')

    plt.plot(db_data_lih_qeuccsd['r'], db_data_lih_qeuccsd['error'], label=r'QUCC', marker='+', linewidth=0.5, color='blue')
    plt.plot(db_data_lih_uccsd['r'], db_data_lih_uccsd['error'], label=r'UCC', marker='+', linewidth=0.5, color='red')

    # plt.vlines([1.546], ymax=200, ymin=-100, linewidth=0.75, color='black', label='ground configuration')
    plt.fill_between([0.25, 3.75], 1e-15, 1e-3, color='lavender', label='chemical accuracy')

    # plt.xlabel(r'Li-H bond distance, $\AA$')
    # plt.ylabel(r'$E-E_{FCI}$, Hartree')
    plt.ylim(1e-5, 1e-1)
    plt.xlim(0.25, 3.25)
    plt.yscale('log')
    plt.grid(b=True, which='major', color='grey', linestyle='--', linewidth=0.5)

    plt.show()

    print('macaroni')