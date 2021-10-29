import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas
from src.q_systems import *
from src.molecules.molecules import *

if __name__ == "__main__":

    lih_g_08_df = pandas.read_csv('../../../../results/dissociation_curves/LiH_h_adapt_gsdqe_e-08_03-Sep-2020.csv')
    lih_exc_1_08 = pandas.read_csv('../../../../results/dissociation_curves/LiH_exc_1_iqeb_08.csv')
    lih_exc_2_08 = pandas.read_csv('../../../../results/dissociation_curves/LiH_exc_2_iqeb_08.csv')
    lih_exc_3_08 = pandas.read_csv('../../../../results/dissociation_curves/LiH_exc_3_iqeb_08.csv')
    lih_exc_4_08 = pandas.read_csv('../../../../results/dissociation_curves/LiH_exc_4_iqeb_08.csv')

    # lih_g_06_df = pandas.read_csv('../../../../results/dissociation_curves/LiH_h_adapt_gsdqe_e-06_02-Sep-2020.csv')
    # lih_g_08_df = pandas.read_csv('../../../../results/dissociation_curves/LiH_h_adapt_gsdqe_e-08_03-Sep-2020.csv')
    #
    # lih_exc_guccsd_df = pandas.read_csv('../../../../results/dissociation_curves/LiH_exc_1_guccsd_03-Apr-2021 (13:40:01.204422).csv')
    # lih_exc_uccsd_df = pandas.read_csv('../../../../results/dissociation_curves/LiH_exc_1_uccsd_04-Apr-2021 (12:20:22.358678).csv')

    col = 'E'

    plt.plot(lih_g_08_df['r'], lih_g_08_df[col], label=r'Ground', marker='+', linewidth=0.5, color='blue')
    plt.plot(lih_exc_1_08['r'], lih_exc_1_08[col], label=r'$1^{st}$ excited', marker='+', linewidth=0.5, color='red')
    plt.plot(lih_exc_2_08['r'], lih_exc_2_08[col], label=r'$2^{nd}$ excited', marker='+', linewidth=0.5, color='green')
    plt.plot(lih_exc_3_08['r'], lih_exc_3_08[col], label=r'$3^{rd}$ excited', marker='+', linewidth=0.2, color='orange')
    plt.plot(lih_exc_4_08['r'], lih_exc_4_08[col], label=r'$4^{th}$ excited', marker='+', linewidth=0.2, color='cyan')

    plt.vlines([1.546], ymax=200, ymin=-100, linewidth=0.75, color='black', label='Equilibrium configuration')
    plt.fill_between([0.5, 3.75], 1e-15, 1e-3, color='lavender', label='chemical accuracy')

    plt.xlabel(r'Li-H bond distance, $\AA$',fontsize=15)
    plt.ylabel('Energy, Hartree',fontsize=15)
    # plt.ylabel(r'$E-E_{FCI}$, Hartree',fontsize=15)

    plt.ylim(-7.9, -7.50)
    # plt.ylim(1e-9, 1e-6)
    # plt.yscale('log')

    plt.xlim(0.75, 3.75)
    plt.grid(b=True, which='major', color='grey', linestyle='--', linewidth=0.5)

    plt.legend(fontsize=12)
    plt.show()

    print('macaroni')