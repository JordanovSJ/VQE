import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

if __name__ == "__main__":

    db_iqeb = pandas.read_csv('../../results/adapt_vqe_results/vip/LiH_h_adapt_gsdqe_comp_pairs_15-Sep-2020.csv')
    db_g_iqeb = pandas.read_csv('../../results/adapt_vqe_results/vip/LiH_g_adapt_gsdqe_comp_pairs_14-Sep-2020.csv')
    db_adapt_pairs = pandas.read_csv('../../results/adapt_vqe_results/vip/LiH_g_adapt_gsdfe_comp_pairs_16-Sep-2020.csv')
    db_adapt = pandas.read_csv('../../results/adapt_vqe_results/vip/LiH_g_adapt_gsdfe_comp_exc_16-Sep-2020.csv')
    db_q_adapt = pandas.read_csv('../../results/adapt_vqe_results/vip/LiH_g_adapt_gsdpwe_15-Sep-2020.csv')

    fig, ax = plt.subplots()

    df_col = 'n'
    linewidth = 0.4
    marker = '_'

    ax.plot(db_iqeb[df_col], db_iqeb['error'], label='IQEB-VQE', marker=marker, linewidth=linewidth, color='blue')
    ax.plot(db_q_adapt[df_col], db_q_adapt['error'], label='qubit-ADAPT-VQE', marker=marker, linewidth=linewidth, color='green')
    ax.plot(db_adapt[df_col], db_adapt['error'], label='ADAPT-VQE', marker=marker, linewidth=linewidth, color='red')
    # ax.plot(db_g_iqeb[df_col], db_g_iqeb['error'], label='grad-IQEB-VQE', marker=marker, linewidth=linewidth)
    # ax.plot(db_adapt_pairs[df_col], db_adapt_pairs['error'], label='pairs-ADAPT', marker=marker, linewidth=linewidth)
    ax.fill_between([0, 11000], 1e-15, 1e-3, color='lavender')

    ax.set_xlabel('Number of iterations')
    ax.set_ylabel(r'$E(\theta) - E_{FCI}$, Hartree')
    ax.set_ylim(1e-9, 1e-2)
    ax.set_xlim(0, 200)
    ax.set_yscale('log')
    ax.grid(b=True, which='major', color='grey', linestyle='--',linewidth=0.5)
    # ax.grid(b=True, which='minor', color='black', linestyle='-.', linewidth=0.3)
    # ax.xaxis.set_minor_locator(MultipleLocator(125))
    # ax.xaxis.set_major_locator(MultipleLocator(250))

    ##  Zoomed
    zoom = 2.25
    zoom_position = 1
    x1, x2, y1, y2 = 0, 25 , 1e-4, 1e-2

    axins = zoomed_inset_axes(ax, zoom, loc=zoom_position)
    axins.plot(db_iqeb[df_col], db_iqeb['error'], label='IQEB-VQE', marker=marker, linewidth=linewidth, color='blue')
    axins.plot(db_q_adapt[df_col], db_q_adapt['error'], label='qubit-ADAPT-VQE', marker=marker, linewidth=linewidth, color='green')
    axins.plot(db_adapt[df_col], db_adapt['error'], label='ADAPT-VQE', marker=marker, linewidth=linewidth, color='red')

    # axins.plot(db_g_iqeb[df_col], db_g_iqeb['error'], label='grad-IQEB-VQE', marker=marker, linewidth=linewidth)
    # axins.plot(db_adapt_pairs[df_col], db_adapt_pairs['error'], label='pairs-ADAPT', marker=marker, linewidth=linewidth)
    axins.fill_between([0, 20000], 1e-9, 1e-3, color='lavender', label='chemical accuracy')
    axins.grid(b=True, which='major', color='black', linestyle='--', linewidth=0.5)
    axins.grid(b=True, which='minor', color='black', linestyle='-.', linewidth=0.2)

    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_yscale('log')
    # axins.xaxis.set_minor_locator(MultipleLocator(125))
    # axins.xaxis.set_major_locator(MultipleLocator(250))

    mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0", linewidth=.75)

    # ax.legend(loc=3)#, bbox_to_anchor=(1,0.4))

    plt.show()

    print('macaroni')
