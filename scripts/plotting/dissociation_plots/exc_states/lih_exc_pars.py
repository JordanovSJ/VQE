import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas
from src.molecular_system import *
from src.molecules.molecules import *

if __name__ == "__main__":

    lih_exc_06_df = pandas.read_csv('../../../../results/dissociation_curves/LiH_exc_1_iqeb_06.csv')
    lih_exc_08_df = pandas.read_csv('../../../../results/dissociation_curves/LiH_exc_1_iqeb_08.csv')

    lih_g_06_df = pandas.read_csv('../../../../results/dissociation_curves/LiH_h_adapt_gsdqe_e-06_02-Sep-2020.csv')
    lih_g_08_df = pandas.read_csv('../../../../results/dissociation_curves/LiH_h_adapt_gsdqe_e-08_03-Sep-2020.csv')

    lih_exc_guccsd_df = pandas.read_csv('../../../../results/dissociation_curves/LiH_exc_1_guccsd_03-Apr-2021 (13:40:01.204422).csv')
    lih_exc_uccsd_df = pandas.read_csv('../../../../results/dissociation_curves/LiH_exc_1_uccsd_04-Apr-2021 (12:20:22.358678).csv')

    col = 'n_iters'

    plt.hlines([200], xmax=1, xmin=3.5, linewidth=0.75, color='red', label=r'$1^{st}$ excited, UCCSD')
    plt.hlines([1521], xmax=1, xmin=3.5, linewidth=0.75, color='magenta', label=r'$1^{st}$ excited, GUCCSD')

    plt.plot(lih_exc_06_df['r'], lih_exc_06_df[col], label=r'$1^{st}$ excited, IQEB-VQE $\epsilon=10^{-6}$ Hartree', marker='+', linewidth=0.5, color='green')
    plt.plot(lih_exc_08_df['r'], lih_exc_08_df[col], label=r'$1^{st}$ excited, IQEB-VQE $\epsilon=10^{-8}$ Hartree', marker='+', linewidth=0.2, color='orange')

    plt.vlines([1.546], ymax=2500, ymin=-100, linewidth=0.75, color='black', label='Equilibrium configuration')
    plt.fill_between([0.5, 3.75], 1e-12, 1e-3, color='lavender', label='chemical accuracy')

    plt.xlabel(r'Li-H bond distance, $\AA$', fontsize=15)
    plt.ylabel('Number of parameters', fontsize=15)
    # plt.ylim(-7.9, -7.65)
    plt.ylim(10, 1600)
    plt.yscale('symlog')

    plt.xlim(0.75, 3.75)
    plt.grid(b=True, which='major', color='grey', linestyle='--', linewidth=0.5)

    plt.yticks([10,25,50,100,250, 500, 1000, 2500], [10,25,50,100,250, 500, 1000, 2500])

    # plt.legend()
    plt.show()

    print('macaroni')