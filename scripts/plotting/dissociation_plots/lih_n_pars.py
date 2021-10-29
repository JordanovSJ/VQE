import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas
from src.q_system import *

if __name__ == "__main__":

    db_data_lih_hf = pandas.read_csv('../../../results/dissociation_curves/LiH_hf_03-Sep-2020.csv')
    db_data_lih_04 = pandas.read_csv('../../../results/dissociation_curves/LiH_iqeb_04.csv')

    db_data_lih_06 = pandas.read_csv('../../../results/dissociation_curves/LiH_h_adapt_gsdqe_e-06_02-Sep-2020.csv')
    db_data_lih_08 = pandas.read_csv('../../../results/dissociation_curves/LiH_h_adapt_gsdqe_e-08_03-Sep-2020.csv')
    db_data_lih_uccsd = pandas.read_csv('../../../results/dissociation_curves/LiH_uccsd_02-Sep-2020.csv')

    plt.plot(db_data_lih_04['r'], db_data_lih_04['n_iters'], label=r'IQEB-VQE $\epsilon=10^{-6}$', marker='+', linewidth=0.5, color='blue')

    plt.plot(db_data_lih_06['r'], db_data_lih_06['n_iters'], label=r'IQEB-VQE $\epsilon=10^{-6}$', marker='+', linewidth=0.5, color='green')
    plt.plot(db_data_lih_08['r'], db_data_lih_08['n_iters'], label=r'IQEB-VQE $\epsilon=10^{-8}$', marker='+', linewidth=0.5, color='red')
    plt.hlines([92],xmin=1, xmax=3.5, color='tab:brown', linewidth=1)

    # plt.vlines([1.546], ymax=200, ymin=-100, linewidth=0.75, color='black', label='ground configuration')
    plt.fill_between([0.5, 3.75], 1e-9, 1e-3, color='lavender', label='chemical accuracy')

    plt.xlabel(r'Li-H bond distance, $\AA$', fontsize=15)
    plt.ylabel(r'Number of ansatz parameters', fontsize=15)
    plt.ylim(0, 95)
    plt.xlim(0.75, 3.75)
    plt.grid(b=True, which='major', color='grey', linestyle='--', linewidth=0.5)

    plt.show()

    print('macaroni')