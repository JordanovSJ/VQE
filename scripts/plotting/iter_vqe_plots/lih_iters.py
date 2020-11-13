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

    db_iqeb = pandas.read_csv('../../../results/iter_vqe_results/vip/LiH_h_adapt_gsdqe_comp_pairs_15-Sep-2020.csv')
    # db_iqeb_1 = pandas.read_csv('../../../results/iter_vqe_results/vip/LiH_h_adapt_gsdqe_comp_pairs_r=1_24-Sep-2020.csv')
    db_iqeb_3 = pandas.read_csv('../../../results/iter_vqe_results/vip/LiH_h_adapt_gsdqe_comp_pair_r=3_24-Sep-2020.csv')

    db_adapt = pandas.read_csv('../../../results/iter_vqe_results/vip/LiH_g_adapt_gsdfe_comp_exc_16-Sep-2020.csv')
    # db_adapt_1 = pandas.read_csv('../../../results/iter_vqe_results/vip/LiH_g_adapt_gsdfe_comp_exc_r=1_25-Sep-2020.csv')
    db_adapt_3 = pandas.read_csv('../../../results/iter_vqe_results/vip/LiH_g_adapt_gsdfe_comp_exc_r=3_30-Oct-2020.csv')

    db_q_adapt = pandas.read_csv('../../../results/iter_vqe_results/vip/LiH_g_adapt_gsdpwe_15-Sep-2020.csv')
    # db_q_adapt_1 = pandas.read_csv('../../../results/iter_vqe_results/vip/LiH_g_adapt_gsdpwe_r=1_21-Sep-2020.csv')
    db_q_adapt_3 = pandas.read_csv('../../../results/iter_vqe_results/vip/LiH_g_adapt_gsdpwe_r=3_24-Sep-2020.csv')

    fig, ax = plt.subplots()

    df_col = 'n'
    linewidth = 0.4
    marker = '_'
    #
    ax.plot(db_iqeb[df_col], db_iqeb['error'], label='IQEB, r=1.546', marker=marker, linewidth=linewidth, color='blue')
    # ax.plot(db_iqeb_1[df_col], db_iqeb_1['error'], label='IQEB, r=1', marker=marker, linewidth=linewidth, color='dodgerblue')
    ax.plot(db_iqeb_3[df_col], db_iqeb_3['error'], label='IQEB, r=3', marker=marker, linewidth=linewidth, color='midnightblue')

    ax.plot(db_q_adapt[df_col], db_q_adapt['error'], label='q-ADAPT, r=1.546', marker=marker, linewidth=linewidth, color='green')
    # ax.plot(db_q_adapt_1[df_col], db_q_adapt_1['error'], label='q-ADAPT, r=1', marker=marker, linewidth=linewidth, color='limegreen')
    ax.plot(db_q_adapt_3[df_col], db_q_adapt_3['error'], label='q-ADAPT, r=3', marker=marker, linewidth=linewidth, color='darkolivegreen')

    ax.plot(db_adapt[df_col], db_adapt['error'], label='ADAPT, r=1.546', marker=marker, linewidth=linewidth, color='red')
    # ax.plot(db_adapt_1[df_col], db_adapt_1['error'], label='ADAPT, r=1', marker=marker, linewidth=linewidth, color='orangered')
    ax.plot(db_adapt_3[df_col], db_adapt_3['error'], label='ADAPT, r=3', marker=marker, linewidth=linewidth, color='darkred')

    ax.fill_between([0, 11000], 1e-15, 1e-3, color='lavender', label='chem. accuracy')

    ax.set_xlabel('Number of iterations')
    ax.set_ylabel(r'$E(\theta) - E_{FCI}$, Hartree')
    ax.set_ylim(1e-9, 1e-1)
    ax.set_xlim(0, 200)
    ax.set_yscale('log')
    ax.grid(b=True, which='major', color='grey', linestyle='--',linewidth=0.5)
    # ax.grid(b=True, which='minor', color='black', linestyle='-.', linewidth=0.3)
    # ax.xaxis.set_minor_locator(MultipleLocator(125))
    # ax.xaxis.set_major_locator(MultipleLocator(250))

    #  Zoomed
    zoom = 1.75
    zoom_position = 1
    x1, x2, y1, y2 = 0, 25 , 1e-4, 1e-1

    axins = zoomed_inset_axes(ax, zoom, loc=zoom_position)
    axins.plot(db_iqeb[df_col], db_iqeb['error'],  marker=marker, linewidth=1.5*linewidth,color='blue')
    # axins.plot(db_iqeb_1[df_col], db_iqeb_1['error'],  marker=marker, linewidth=1.5*linewidth,color='dodgerblue')
    axins.plot(db_iqeb_3[df_col], db_iqeb_3['error'], marker=marker, linewidth=1.5*linewidth,color='midnightblue')

    axins.plot(db_q_adapt[df_col], db_q_adapt['error'], label='qubit-ADAPT-VQE, r=1.546', marker=marker,linewidth=linewidth, color='green')
    # axins.plot(db_q_adapt_1[df_col], db_q_adapt_1['error'], marker=marker,linewidth=1.5*linewidth, color='limegreen')
    axins.plot(db_q_adapt_3[df_col], db_q_adapt_3['error'], marker=marker,linewidth=1.5*linewidth, color='darkolivegreen')

    axins.plot(db_adapt[df_col], db_adapt['error'], marker=marker, linewidth=1.5*linewidth,color='red')
    # axins.plot(db_adapt_1[df_col], db_adapt_1['error'], marker=marker, linewidth=1.5*linewidth,color='orangered')
    axins.plot(db_adapt_3[df_col], db_adapt_3['error'], marker=marker, linewidth=1.5*linewidth,color='darkred')

    axins.fill_between([0, 20000], 1e-9, 1e-3, color='lavender', label='chemical accuracy')
    axins.grid(b=True, which='major', color='black', linestyle='--', linewidth=0.5)
    axins.grid(b=True, which='minor', color='black', linestyle='-.', linewidth=0.2)

    axins.set_xlim(x1, x2)
    axins.xaxis.tick_top()
    axins.set_ylim(y1, y2)
    axins.yaxis.tick_right()
    axins.set_yscale('log')
    # axins.xaxis.set_minor_locator(MultipleLocator(125))
    # axins.xaxis.set_major_locator(MultipleLocator(250))

    mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0", linewidth=.75)

    # ax.legend(loc=3)#, bbox_to_anchor=(1,0.4))
    ax.legend(loc=9, fontsize=10)#, bbox_to_anchor=(1,0.4))

    plt.show()

    print('macaroni')
