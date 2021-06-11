import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas
import numpy

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

if __name__ == "__main__":

    db_n_1_bh2 = pandas.read_csv('../../../results/iter_vqe_results/BeH2_iqeb_n=1_gsdqe_r=3_05-Dec-2020.csv')
    db_n_5_bh2 = pandas.read_csv('../../../results/iter_vqe_results/BeH2_iqeb_q_exc_n=5_r=3_17-Mar-2021.csv')
    db_n_10_bh2 = pandas.read_csv('../../../results/iter_vqe_results/vip/BeH2_h_adapt_gsdqe_comp_pair_r=3_06-Oct-2020.csv')
    # db_n_10_bh2 = pandas.read_csv('../../../results/iter_vqe_results/BeH2_iqeb_n=10_gsdqe_r=3_02-Dec-2020.csv')
    db_n_20_bh2 = pandas.read_csv('../../../results/iter_vqe_results/BeH2_iqeb_n=20_gsdqe_r=3_02-Dec-2020.csv')
    db_n_30_bh2 = pandas.read_csv('../../../results/iter_vqe_results/BeH2_iqeb_n=30_gsdqe_r=3_02-Dec-2020.csv')

    db_n_1_lih = pandas.read_csv('../../../results/iter_vqe_results/LiH_iqeb_n=1_gsdqe_r=3_03-Dec-2020.csv')
    db_n_5_lih = pandas.read_csv('../../../results/iter_vqe_results/LiH_iqeb_q_exc_n=5_r=3_17-Mar-2021.csv')
    # db_n_10_lih = pandas.read_csv('../../../results/iter_vqe_results/vip/LiH_h_adapt_gsdqe_comp_pair_r=3_24-Sep-2020.csv')
    db_n_10_lih = pandas.read_csv('../../../results/iter_vqe_results/LiH_iqeb_q_exc_n=10_r=3_17-Mar-2021.csv')
    db_n_30_lih = pandas.read_csv('../../../results/iter_vqe_results/LiH_iqeb_n=30_gsdqe_r=3-03-Dec-2020.csv')
    # db_n_30_lih = pandas.read_csv('../../../results/iter_vqe_results/LiH_iqeb_q_exc_n=30_r=3_17-Mar-2021.csv')

    db_n_1_h6 = pandas.read_csv('../../../results/iter_vqe_results/H6_iqeb_q_exc_n=1_r=3_30-May-2021.csv')
    db_n_10_h6 = pandas.read_csv('../../../results/iter_vqe_results/H6_iqeb_q_exc_n=10_r=3_SDEexc_27-May-2021.csv')
    db_n_5_h6 = pandas.read_csv('../../../results/iter_vqe_results/H6_iqeb_q_exc_n=5_r=3_03-Jun-2021.csv')
    db_n_30_h6 = pandas.read_csv('../../../results/iter_vqe_results/H6_iqeb_q_exc_n=30_r=3_03-Jun-2021.csv')

    fig, ax = plt.subplots()

    df_col = 'cnot_count'
    linewidth = 0.4
    marker = '_'

    # ax.plot(db_n_1_bh2[df_col], db_n_1_bh2['error'], label=r'BeH$_2$, $n_{QE}$=1', marker=marker, linewidth=linewidth, color='red')
    # ax.plot(db_n_5_bh2[df_col], db_n_5_bh2['error'], label=r'BeH$_2$, $n_{QE}$=5', marker=marker, linewidth=linewidth, color='yellow')
    # ax.plot(db_n_10_bh2[df_col], db_n_10_bh2['error'], label=r'BeH$_2$, $n_{QE}$=10', marker=marker, linewidth=linewidth, color='blue')
    # # ax.plot(db_n_20_bh2[df_col], db_n_20_bh2['error'], label='n=20', marker=marker, linewidth=linewidth, color='cornflowerblue')
    # ax.plot(db_n_30_bh2[df_col], db_n_30_bh2['error'], label=r'BeH$_2$, $n_{QE}$=30', marker=marker, linewidth=linewidth, color='green')

    # ax.plot(db_n_1_lih[df_col], db_n_1_lih['error'], label='LiH, $n_{QE}$=1', marker=marker, linewidth=linewidth, color='red')
    # ax.plot(db_n_5_lih[df_col], db_n_5_lih['error'], label='LiH, $n_{QE}$=5', marker=marker, linewidth=linewidth, color='yellow')
    # ax.plot(db_n_10_lih[df_col], db_n_10_lih['error'], label='LiH, $n_{QE}$=10', marker=marker, linewidth=linewidth, color='blue')
    # ax.plot(db_n_30_lih[df_col], db_n_30_lih['error'], label='LiH, $n_{QE}$=30', marker=marker, linewidth=linewidth, color='green')

    ax.plot(db_n_1_h6[df_col], db_n_1_h6['error'], label='H6, n=1', marker=marker, linewidth=linewidth, color='red')
    ax.plot(db_n_5_h6[df_col], db_n_5_h6['error'], label='H6, n=5', marker=marker, linewidth=linewidth, color='yellow')
    ax.plot(db_n_10_h6[df_col], db_n_10_h6['error'], label='H6, n=10', marker=marker, linewidth=linewidth, color='blue')
    ax.plot(db_n_30_h6[df_col], db_n_30_h6['error'], label='H6, n=30', marker=marker, linewidth=linewidth, color='green')

    ax.fill_between([0, 11000], 1e-15, 1e-3, color='lavender', label='chemical accuracy')

    ax.set_xlabel('Number of CNOTs', fontsize=15)
    # ax.set_ylabel(r'$E(\theta) - E_{FCI}$, Hartree', fontsize=15)
    ax.set_ylim(1e-13, 1e-0)
    # ax.set_xlim(0, 250)
    # ax.set_xlim(0, 800)
    ax.set_xlim(0, 2500)
    ax.set_yscale('log')
    ax.grid(b=True, which='major', color='grey', linestyle='--',linewidth=0.5)

    #  Zoomed
    # zoom = 1.6
    # zoom_position = 1
    # x1, x2, y1, y2 = 0, 300, 1e-4, 0.2
    #
    # axins = zoomed_inset_axes(ax, zoom, loc=zoom_position)
    # axins.plot(db_n_1[df_col], db_n_1['error'], label='n=1', marker=marker, linewidth=linewidth, color='forestgreen')
    # axins.plot(db_n_10[df_col], db_n_10['error'], label='n=10', marker=marker, linewidth=linewidth, color='orange')
    # axins.plot(db_n_20[df_col], db_n_20['error'], label='n=20', marker=marker, linewidth=linewidth, color='cornflowerblue')
    #
    # axins.fill_between([0, 20000], 1e-9, 1e-3, color='lavender', label='chemical accuracy')
    # axins.grid(b=True, which='major', color='black', linestyle='--', linewidth=0.5)
    # axins.grid(b=True, which='minor', color='black', linestyle='-.', linewidth=0.2)
    #
    # axins.set_xlim(x1, x2)
    # axins.xaxis.tick_top()
    # axins.set_ylim(y1, y2)
    # axins.yaxis.tick_right()
    # axins.set_yscale('log')

    ax.legend(loc=3, fontsize=15)#, bbox_to_anchor=(1,0.4))

    plt.show()

    print('macaroni')
