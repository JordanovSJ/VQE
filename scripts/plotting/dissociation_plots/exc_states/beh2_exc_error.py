import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas
from src.q_systems import *
from src.molecules.molecules import *

if __name__ == "__main__":

    exc_06_df = pandas.read_csv('../../../../results/dissociation_curves/BeH2_exc_1_iqeb_06.csv')
    exc_08_df = pandas.read_csv('../../../../results/dissociation_curves/BeH2_exc_1_iqeb_08.csv')

    g_06_df = pandas.read_csv('../../../../results/dissociation_curves/BeH2_h_adapt_gsdqe_e-06_03-Sep-2020.csv')

    exc_guccsd_df = pandas.read_csv('../../../../results/dissociation_curves/BeH2_exc_1_guccsd_04-Apr-2021.csv')
    exc_uccsd_df = pandas.read_csv('../../../../results/dissociation_curves/BeH2_exc_1_uccsd_04-Apr-2021.csv')

    col = 'error'

    plt.plot(exc_uccsd_df['r'], exc_uccsd_df[col], label=r'$1^{st}$ excited, UCCSD', marker='+', linewidth=0.5, color='red')
    plt.plot(exc_guccsd_df['r'], exc_guccsd_df[col], label=r'$1^{st}$ excited, GUCCSD', marker='+', linewidth=0.5, color='magenta', alpha=0.5)
    plt.plot(exc_06_df['r'], exc_06_df[col], label=r'$1^{st}$ excited, IQEB-VQE $\epsilon=10^{-6}$ Hartree', marker='+', linewidth=0.5, color='green')
    plt.plot(exc_08_df['r'], exc_08_df[col], label=r'$1^{st}$ excited, IQEB-VQE $\epsilon=10^{-8}$ Hartree', marker='+', linewidth=0.2, color='orange')

    # plt.plot(lih_g_08_df['r'], lih_g_08_df[col], label=r'Ground state, IQEB-VQE $\epsilon=10^{-8}$ Hartree', marker='+', linewidth=0.5, color='red', alpha=0.5)

    plt.vlines([1.316], ymax=200, ymin=-100, linewidth=0.75, color='black', label='Equilibrium configuration')
    plt.fill_between([0.5, 3.75], 1e-12, 1e-3, color='lavender', label='chemical accuracy')

    # plt.xlabel(r'Be-H bond distance, $\AA$')
    # plt.ylabel(r'$E-E_{FCI}$, Hartree')
    # plt.ylim(-15.6, -14.7)
    plt.ylim(1e-8, 1e-0)
    plt.yscale('log')

    plt.xlim(0.5, 3.25)
    plt.grid(b=True, which='major', color='grey', linestyle='--', linewidth=0.5)

    # plt.legend()
    plt.show()

    print('macaroni')