import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas
from src.molecular_system import *

if __name__ == "__main__":
    db_data_lih_hf = pandas.read_csv('../../../results/dissociation_curves/BeH2_hf_06-Oct-2020.csv')
    # db_data_lih_06 = pandas.read_csv('../../../results/dissociation_curves/BeH2_h_adapt_gsdqe_e-06_03-Sep-2020.csv')
    # db_data_lih_08 = pandas.read_csv('../../../results/dissociation_curves/BeH2_h_adapt_gsdqe_e-08_15-Sep-2020.csv')

    db_data_lih_06 = pandas.read_csv('../../../results/dissociation_curves/BeH2_iqeb_06_new.csv')
    db_data_lih_08 = pandas.read_csv('../../../results/dissociation_curves/BeH2_iqeb_08_new.csv')
    db_data_lih_uccsd = pandas.read_csv('../../../results/dissociation_curves/BeH2_uccsd_08-Sep-2020.csv')

    plt.plot(db_data_lih_06['r'], db_data_lih_06['n_iters'], label=r'IQEB-VQE $\epsilon=10^{-6}$', marker='+', linewidth=0.5, color='green')
    plt.plot(db_data_lih_08['r'], db_data_lih_08['n_iters'], label=r'IQEB-VQE $\epsilon=10^{-8}$', marker='+', linewidth=0.5, color='red')
    plt.hlines([204],xmin=0.75, xmax=3.25, color='orange', linewidth=0.75)

    plt.vlines([1.316], ymax=220, ymin=-100, linewidth=0.75, color='black', label='ground configuration')
    plt.fill_between([0.5, 3.75], 1e-9, 1e-3, color='lavender', label='chemical accuracy')

    plt.xlabel(r'Be-H bond distance, $\AA$')
    plt.ylabel(r'Number of ansatz parameters')
    plt.ylim(30, 210)
    plt.xlim(0.5, 3.5)
    plt.grid(b=True, which='major', color='grey', linestyle='--', linewidth=0.5)

    plt.show()

    print('macaroni')