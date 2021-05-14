import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas
from src.molecular_system import *

if __name__ == "__main__":

    db_data_beh2_qeuccsd = pandas.read_csv('../../../results/dissociation_curves/BeH2_qeuccsd_08-Mar-2021.csv')
    db_data_beh2_uccsd = pandas.read_csv('../../../results/dissociation_curves/BeH2_uccsd_08-Sep-2020.csv')

    plt.plot(db_data_beh2_qeuccsd['r'], db_data_beh2_qeuccsd['error'], label=r'QUCC', marker='+', linewidth=0.5, color='blue')
    plt.plot(db_data_beh2_uccsd['r'], db_data_beh2_uccsd['error'], label=r'UCC', marker='+', linewidth=0.5, color='red')

    plt.vlines([1.316], ymax=200, ymin=-100, linewidth=0.75, color='black', label='ground configuration')
    plt.fill_between([0.5, 3.75], 1e-9, 1e-3, color='lavender', label='chemical accuracy')

    plt.xlabel(r'Be-H bond distance, $\AA$')
    plt.ylabel(r'$E-E_{FCI}$, Hartree')
    plt.ylim(1e-4, 1e-1)
    plt.xlim(0.5, 3.5)
    plt.yscale('log')
    plt.grid(b=True, which='major', color='grey', linestyle='--', linewidth=0.5)

    plt.show()

    print('macaroni')