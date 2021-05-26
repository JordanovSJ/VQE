import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas
from src.q_system import *
from src.molecules.molecules import *
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


if __name__ == "__main__":

    db_data_lih_06 = pandas.read_csv('../../../../results/dissociation_curves/BeH2_exc_1_iqeb_06.csv')
    # db_data_lih_08 = pandas.read_csv('../../../../results/dissociation_curves/BeH2_exc_1_iqeb_08.csv')
    db_data_lih_08 = pandas.read_csv('../../../../results/dissociation_curves/BeH2_iqeb_08_new.csv')

    df_fcis = pandas.read_csv('../../../../results/dissociation_curves/BeH2_exc_1_fci_E.csv')
    df_fcis_2 = pandas.read_csv('../../../../results/dissociation_curves/BeH2_exc_2_fci_E.csv')
    df_fcis_3 = pandas.read_csv('../../../../results/dissociation_curves/BeH2_exc_3_fci_E.csv')
    df_fcis_4 = pandas.read_csv('../../../../results/dissociation_curves/BeH2_exc_4_fci_E.csv')
    df_fcis_5 = pandas.read_csv('../../../../results/dissociation_curves/BeH2_exc_5_fci_E.csv')
    df_fcis_6 = pandas.read_csv('../../../../results/dissociation_curves/BeH2_exc_6_fci_E.csv')
    df_fcis_7 = pandas.read_csv('../../../../results/dissociation_curves/BeH2_exc_7_fci_E.csv')
    df_fcis_8 = pandas.read_csv('../../../../results/dissociation_curves/BeH2_exc_8_fci_E.csv')
    df_fcis_9 = pandas.read_csv('../../../../results/dissociation_curves/BeH2_exc_9_fci_E.csv')

    # db_data_lih_uccsd = pandas.read_csv('../../../results/dissociation_curves/BeH2_uccsd_08-Sep-2020.csv')

    col = 'error'
    marker = '*'
    size = 3

    fig, ax = plt.subplots()

    # fci_Es = []
    # fci_rs = []
    # for i in range(30):
    #     r = 0.75 + 2.25*i/30
    #     fci_rs.append(r)
    #     fci_Es.append(BeH2(r=r).fci_energy)
    #
    # print(fci_rs)
    # print(fci_Es)
    fci_rs = [0.75, 0.825, 0.9, 0.975, 1.05, 1.125, 1.2, 1.275, 1.35, 1.425, 1.5, 1.575, 1.65, 1.725, 1.8, 1.875, 1.95, 2.025, 2.1, 2.175, 2.25, 2.325, 2.4, 2.475, 2.55, 2.625, 2.7, 2.775, 2.85, 2.925]
    fci_Es = [-15.03872303015795, -15.229289360040283, -15.364150937727837, -15.45777581456385, -15.520754073707359, -15.560850275285127, -15.583812048966928, -15.593938638821541, -15.594473178337996, -15.587880558297142, -15.576051245101569, -15.560453185867859, -15.542244319720126, -15.522354596903526, -15.501545086154817, -15.480450643440934, -15.459611226304848, -15.439495453327698, -15.420518550926047, -15.403055084013786, -15.38744402222435, -15.37397915639995, -15.362874312619507, -15.354201188393517, -15.347829636246441, -15.34342956459782, -15.340557704299998, -15.338773539697097, -15.337714942778776, -15.337118028755619]

    # #
    # # # plt.plot(db_data_lih_hf['r'], db_data_lih_hf['E'], label='HF', marker='+', linewidth=0.5)
    # # # plt.plot(db_data_lih_uccsd['r'], db_data_lih_uccsd['E'], label=r'UCCSD', marker='+', linewidth=0.5)
    # # plt.plot(db_data_lih_06['r'], db_data_lih_06[col], label=r'IQEB-VQE $\epsilon=10^{-6}$ Hartree', marker='+', linewidth=0.5, color='green')
    ax.plot(fci_rs, fci_Es, label=r'ground state', linewidth=0.5, color='black')
    ax.plot(df_fcis['r'], df_fcis['fci_E'], label='exc. st. 1', color='purple', linewidth=0.5, alpha=0.75)
    ax.plot(df_fcis_2['r'], df_fcis_2['fci_E'], label='exc. st. 2', color='cyan', linewidth=0.5, alpha=0.75)
    ax.plot(df_fcis_3['r'], df_fcis_3['fci_E'], label='exc. st. 3', color='orange', linewidth=0.5, alpha=0.75)
    ax.plot(df_fcis_4['r'], df_fcis_4['fci_E'], label='exc. st. 4', color='blue', linewidth=0.5, alpha=0.75)
    ax.plot(df_fcis_5['r'], df_fcis_5['fci_E'], label='exc. st. 5', color='green', linewidth=0.5, alpha=0.75)
    ax.plot(df_fcis_6['r'], df_fcis_6['fci_E'], label='exc. st. 6', color='yellow', linewidth=0.5, alpha=0.75)
    ax.plot(df_fcis_7['r'], df_fcis_7['fci_E'], label='exc. st. 7', color='magenta', linewidth=0.5, alpha=0.75)
    ax.plot(df_fcis_8['r'], df_fcis_8['fci_E'], label='exc. st. 8', color='darkblue', linewidth=0.5, alpha=0.75)
    ax.plot(df_fcis_9['r'], df_fcis_9['fci_E'], label='exc. st. 9', color='red', linewidth=0.5, alpha=0.75)

    # ax.vlines([1.316], ymax=200, ymin=-100, linewidth=0.75, color='black', label='equilibrium config.')
    # ax.fill_between([0.5, 3.75], 1e-9, 1e-3, color='lavender', label='chem. accuracy')

    ax.set_xlabel(r'Be-H bond distance, $\AA$')
    ax.set_ylabel('Energy, Hartree')
    ax.set_ylim(-15.6, -14.7)
    # plt.ylim(1e-9, 1e-0)
    # plt.yscale('log')

    ax.set_xlim(0.5, 3.25)
    ax.grid(b=True, which='major', color='grey', linestyle='--', linewidth=0.5)


    zoom = 5
    zoom_position = 1
    x1, x2, y1, y2 = 1.4, 1.8, -15.34,  -15.28
    axins = zoomed_inset_axes(ax, zoom, loc=zoom_position)

    axins.plot(fci_rs, fci_Es, label=r'ground state', linewidth=0.5, color='black')
    axins.plot(df_fcis['r'], df_fcis['fci_E'], marker=marker, markersize=size, label='exc. st. 1', color='purple', linewidth=0.25, alpha=0.5)
    axins.plot(df_fcis_2['r'], df_fcis_2['fci_E'], marker=marker, markersize=size, label='exc. st. 2', color='cyan', linewidth=0.25, alpha=0.5)
    axins.plot(df_fcis_3['r'], df_fcis_3['fci_E'],marker=marker, markersize=size, label='exc. st. 3', color='orange', linewidth=0.25, alpha=0.75)
    axins.plot(df_fcis_4['r'], df_fcis_4['fci_E'],marker=marker, markersize=size, label='exc. st. 4', color='blue', linewidth=0.25, alpha=0.5)
    axins.plot(df_fcis_5['r'], df_fcis_5['fci_E'],marker=marker, markersize=size, label='exc. st. 5', color='green', linewidth=0.25, alpha=0.5)
    axins.plot(df_fcis_6['r'], df_fcis_6['fci_E'],marker=marker, markersize=size, label='exc. st. 6', color='yellow', linewidth=0.25, alpha=0.5)
    axins.plot(df_fcis_7['r'], df_fcis_7['fci_E'],marker=marker, markersize=size, label='exc. st. 7', color='magenta', linewidth=0.25, alpha=0.5)
    axins.plot(df_fcis_8['r'], df_fcis_8['fci_E'],marker=marker, markersize=size, label='exc. st. 8', color='darkblue', linewidth=0.25, alpha=0.5)
    axins.plot(df_fcis_9['r'], df_fcis_9['fci_E'],marker=marker, markersize=size, label='exc. st. 9', color='red', linewidth=0.25, alpha=0.75)

    axins.grid(b=True, which='major', color='black', linestyle='--', linewidth=0.5)
    axins.grid(b=True, which='minor', color='black', linestyle='-.', linewidth=0.2)
    axins.set_xlim(x1, x2)
    axins.xaxis.tick_top()
    axins.set_ylim(y1, y2)
    # axins.set_yscale('log')

    mark_inset(ax, axins, loc1=4, loc2=3, fc="none", ec="0", linewidth=.5)

    ax.legend(bbox_to_anchor=(1.3, 1.05))
    plt.show()

    print('macaroni')