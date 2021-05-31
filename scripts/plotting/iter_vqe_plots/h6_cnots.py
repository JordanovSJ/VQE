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

    # db_iqeb = pandas.read_csv('../../../results/iter_vqe_results/H6_iqeb_q_exc_n=10_r=15_27-May-2021.csv')
    db_iqeb = pandas.read_csv('../../../results/iter_vqe_results/H6_iqeb_q_exc_n=1_r=15_27-May-2021.csv')
    db_iqeb_3 = pandas.read_csv('../../../results/iter_vqe_results/H6_iqeb_q_exc_n=1_r=3_30-May-2021.csv')

    db_adapt = pandas.read_csv('../../../results/iter_vqe_results/H6_adapt_r=15_27-May-2021.csv')
    db_adapt_3 = pandas.read_csv('../../../results/iter_vqe_results/H6_adapt_r=3_SDEexc_27-May-2021.csv')

    db_q_adapt = pandas.read_csv('../../../results/iter_vqe_results/H6_q_adapt_r=15_complete_27-May-2021.csv')
    db_q_adapt_3 = pandas.read_csv('../../../results/iter_vqe_results/H6_q_adapt_r=3_SDEexc_27-May-2021.csv')

    fig, ax = plt.subplots()

    df_col = 'cnot_count'
    linewidth = 0.5
    marker = '_'
    #
    ax.plot(db_iqeb[df_col], db_iqeb['error'], label=r'IQEB, $1.546\AA$', marker=marker, linewidth=linewidth, color='blue')
    ax.plot(db_iqeb_3[df_col], db_iqeb_3['error'], label=r'IQEB, $3\AA$', marker=marker, linewidth=linewidth, color='midnightblue')

    ax.plot(db_q_adapt[df_col], db_q_adapt['error'], label=r'q-ADAPT, $1.546\AA$', marker=marker, linewidth=linewidth, color='green')
    ax.plot(db_q_adapt_3[df_col], db_q_adapt_3['error'], label=r'q-ADAPT, $3\AA$', marker=marker, linewidth=linewidth, color='darkolivegreen')

    ax.plot(db_adapt[df_col], db_adapt['error'], label=r'ADAPT, $1.546\AA$', marker=marker, linewidth=linewidth, color='red')
    ax.plot(db_adapt_3[df_col], db_adapt_3['error'], label=r'ADAPT, $r=3\AA$', marker=marker, linewidth=linewidth, color='darkred')

    ax.fill_between([0, 11000], 1e-15, 1e-3, color='lavender', label='chem. accuracy')

    ax.set_xlabel('Number of CNOTs')
    ax.set_ylabel(r'$E(\theta) - E_{FCI}$, Hartree')
    ax.set_ylim(1e-9, 1)
    ax.set_xlim(0, 6000)
    ax.set_yscale('log')
    ax.grid(b=True, which='major', color='grey', linestyle='--',linewidth=0.5)
    # ax.grid(b=True, which='minor', color='black', linestyle='-.', linewidth=0.3)
    # ax.xaxis.set_minor_locator(MultipleLocator(125))
    # ax.xaxis.set_major_locator(MultipleLocator(250))

    #  Zoomed
    zoom = 1.6
    zoom_position = 1
    x1, x2, y1, y2 = 0, 100 , 1e-4, 1e-2

    axins = zoomed_inset_axes(ax, zoom, loc=zoom_position)
    axins.plot(db_iqeb[df_col], db_iqeb['error'],  marker=marker, linewidth=1.5*linewidth,color='blue')
    axins.plot(db_iqeb_3[df_col], db_iqeb_3['error'], marker=marker, linewidth=1.5*linewidth,color='midnightblue')

    axins.plot(db_q_adapt[df_col], db_q_adapt['error'], label='qubit-ADAPT-VQE, r=1.546', marker=marker,linewidth=linewidth, color='green')
    axins.plot(db_q_adapt_3[df_col], db_q_adapt_3['error'], marker=marker,linewidth=1.5*linewidth, color='darkolivegreen')

    axins.plot(db_adapt[df_col], db_adapt['error'], marker=marker, linewidth=1.5*linewidth,color='red')
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

    plt.show()

    print('macaroni')
