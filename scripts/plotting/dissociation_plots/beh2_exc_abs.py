import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas
from src.q_system import *
from src.molecules.molecules import *

if __name__ == "__main__":

    db_data_lih_06 = pandas.read_csv('../../../results/dissociation_curves/BeH2_exc_1_iqeb_06.csv')
    # db_data_lih_08 = pandas.read_csv('../../../results/dissociation_curves/BeH2_exc_1_iqeb_08.csv')
    db_data_lih_08 = pandas.read_csv('../../../results/dissociation_curves/BeH2_iqeb_08_new.csv')

    df_fcis = pandas.read_csv('../../../results/dissociation_curves/BeH2_exc_1_fci_E.csv')
    df_fcis_2 = pandas.read_csv('../../../results/dissociation_curves/BeH2_exc_2_fci_E.csv')
    df_fcis_3 = pandas.read_csv('../../../results/dissociation_curves/BeH2_exc_3_fci_E.csv')
    df_fcis_4 = pandas.read_csv('../../../results/dissociation_curves/BeH2_exc_4_fci_E.csv')
    df_fcis_5 = pandas.read_csv('../../../results/dissociation_curves/BeH2_exc_5_fci_E.csv')
    df_fcis_6 = pandas.read_csv('../../../results/dissociation_curves/BeH2_exc_6_fci_E.csv')
    df_fcis_7 = pandas.read_csv('../../../results/dissociation_curves/BeH2_exc_7_fci_E.csv')
    df_fcis_8 = pandas.read_csv('../../../results/dissociation_curves/BeH2_exc_8_fci_E.csv')
    df_fcis_9 = pandas.read_csv('../../../results/dissociation_curves/BeH2_exc_9_fci_E.csv')

    # db_data_lih_uccsd = pandas.read_csv('../../../results/dissociation_curves/BeH2_uccsd_08-Sep-2020.csv')

    # fci_Es = []
    # fci_rs = []
    # for i in range(30):
    #     r = 1 + 2.5*i/30
    #     fci_rs.append(r)
    #     fci_Es.append(BeH2(r=r).fci_energy)
    #
    # plt.plot(db_data_lih_hf['r'], db_data_lih_hf['E'], label='HF', marker='+', linewidth=0.5)
    # plt.plot(db_data_lih_uccsd['r'], db_data_lih_uccsd['E'], label=r'UCCSD', marker='+', linewidth=0.5)
    # plt.plot(db_data_lih_06['r'], db_data_lih_06['E'], label=r'IQEB-VQE $\epsilon=10^{-6}$ Hartree', marker='+', linewidth=0.5, color='green')
    # plt.plot(db_data_lih_08['r'], db_data_lih_08['E'], label=r'IQEB-VQE $\epsilon=10^{-8}$ Hartree', marker='+', linewidth=0.5, color='red')
    plt.plot(df_fcis['r'], df_fcis['fci_E'], label='FCI energy', color='purple')
    plt.plot(df_fcis_2['r'], df_fcis_2['fci_E'], label='FCI energy 2', color='red')
    plt.plot(df_fcis_3['r'], df_fcis_3['fci_E'], label='FCI energy 3', color='yellow')
    plt.plot(df_fcis_4['r'], df_fcis_4['fci_E'], label='FCI energy 4', color='blue')
    plt.plot(df_fcis_5['r'], df_fcis_5['fci_E'], label='FCI energy 5', color='green')
    # plt.plot(df_fcis_6['r'], df_fcis_6['fci_E'], label='FCI energy 6', color='orange')
    # plt.plot(df_fcis_7['r'], df_fcis_7['fci_E'], label='FCI energy 7', color='orange')
    # plt.plot(df_fcis_8['r'], df_fcis_8['fci_E'], label='FCI energy 8', color='orange')
    plt.plot(df_fcis_9['r'], df_fcis_9['fci_E'], label='FCI energy 9', color='orange')


    plt.vlines([1.316], ymax=200, ymin=-100, linewidth=0.75, color='black', label='ground configuration')
    plt.fill_between([0.5, 3.75], 1e-9, 1e-3, color='lavender', label='chemical accuracy')

    plt.xlabel(r'Be-H bond distance, $\AA$')
    plt.ylabel('Energy, Hartree')
    plt.ylim(-15.6, -14.6)
    plt.xlim(0.5, 3.25)
    plt.grid(b=True, which='major', color='grey', linestyle='--', linewidth=0.5)

    plt.legend()
    plt.show()

    print('macaroni')