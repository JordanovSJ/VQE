import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas
from src.q_systems import *
from src.molecules.molecules import *

if __name__ == "__main__":

    lih_exc_06_df = pandas.read_csv('../../../../results/dissociation_curves/LiH_exc_1_iqeb_06.csv')
    lih_exc_08_df = pandas.read_csv('../../../../results/dissociation_curves/LiH_exc_1_iqeb_08.csv')

    lih_g_06_df = pandas.read_csv('../../../../results/dissociation_curves/LiH_h_adapt_gsdqe_e-06_02-Sep-2020.csv')
    lih_g_08_df = pandas.read_csv('../../../../results/dissociation_curves/LiH_h_adapt_gsdqe_e-08_03-Sep-2020.csv')

    lih_exc_guccsd_df = pandas.read_csv('../../../../results/dissociation_curves/LiH_exc_1_guccsd_03-Apr-2021 (13:40:01.204422).csv')
    lih_exc_uccsd_df = pandas.read_csv('../../../../results/dissociation_curves/LiH_exc_1_uccsd_04-Apr-2021 (12:20:22.358678).csv')

    col = 'E'

    plt.plot(lih_exc_uccsd_df['r'], lih_exc_uccsd_df[col], label=r'$1^{st}$ excited, UCCSD', marker='+', linewidth=0.5, color='red')
    plt.plot(lih_exc_guccsd_df['r'], lih_exc_guccsd_df[col], label=r'$1^{st}$ excited, GUCCSD', marker='+', linewidth=0.5, color='magenta', alpha=0.5)
    plt.plot(lih_exc_08_df['r'], lih_exc_08_df['fci_E'], label=r'$1^{st}$ excited, FCI energy', marker='+', linewidth=0.5, color='darkblue')
    plt.plot(lih_exc_06_df['r'], lih_exc_06_df[col], label=r'$1^{st}$ excited, QEB-ADAPT $\epsilon=10^{-6}$', marker='+', linewidth=0.5, color='green')
    plt.plot(lih_exc_08_df['r'], lih_exc_08_df[col], label=r'$1^{st}$ excited, QEB-ADAPT $\epsilon=10^{-8}$', marker='+', linewidth=0.2, color='orange')

    plt.plot(lih_g_06_df['r'], lih_g_06_df['fci_E'], label=r'Ground state FCI energy', marker='+', linewidth=0.5, color='blue')
    # plt.plot(lih_g_08_df['r'], lih_g_08_df[col], label=r'Ground state, IQEB-VQE $\epsilon=10^{-8}$ Hartree', marker='+', linewidth=0.5, color='red', alpha=0.5)

    plt.vlines([1.546], ymax=200, ymin=-100, linewidth=0.75, color='black', label='Equilibrium configuration')
    plt.fill_between([0.5, 3.75], 1e-12, 1e-3, color='lavender', label='chemical accuracy')

    # plt.xlabel(r'Li-H bond distance, $\AA$')
    plt.ylabel('Energy, Hartree', fontsize=15)
    plt.ylim(-7.9, -7.65)
    # plt.ylim(1e-12, 1e-0)
    # plt.yscale('log')

    plt.xlim(0.75, 3.75)
    plt.grid(b=True, which='major', color='grey', linestyle='--', linewidth=0.5)

    plt.legend()
    plt.show()

    print('macaroni')