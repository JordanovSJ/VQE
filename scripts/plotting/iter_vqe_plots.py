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

    db_h_qe = pandas.read_csv('../../results/adapt_vqe_results/vip/BeH2_h_adapt_gsdqe_13-Aug-2020.csv')
    db_g_pwe = pandas.read_csv('../../results/adapt_vqe_results/vip/BeH2_g_adapt_gsdpwe_restricted_combinations_25-Aug-2020.csv')
    db_g_efe = pandas.read_csv('../../results/adapt_vqe_results/vip/BeH2_g_adapt_gsdefe_corrected_09-Sep-2020.csv')
    db_g_qe = pandas.read_csv('../../results/adapt_vqe_results/vip/BeH2_g_adapt_gsdqe_27-Aug-2020.csv')

    fig, ax = plt.subplots()

    ax.plot(db_h_qe['cnot_count'], db_h_qe['error'], label='IQEB-VQE')
    ax.plot(db_g_qe['cnot_count'], db_g_qe['error'], label='grad-IQEB-VQE')
    ax.plot(db_g_efe['cnot_count'], db_g_efe['error'], label='ADAPT-VQE')
    ax.plot(db_g_pwe['cnot_count'], db_g_pwe['error'], label='qubit-ADAPT-VQE')
    ax.fill_between([0, 11000], 1e-9, 1e-3, color='lavender')

    ax.set_xlabel('Number of CNOT gates')
    ax.set_ylabel(r'$E(\theta) - E_{FCI}$, Hartree')
    ax.set_ylim(1e-7, 1e-1)
    ax.set_xlim(0, 2000)
    ax.set_yscale('log')
    ax.grid(b=True, which='major', color='grey', linestyle='--',linewidth=0.5)
    # ax.grid(b=True, which='minor', color='black', linestyle='-.', linewidth=0.3)
    ax.xaxis.set_minor_locator(MultipleLocator(125))
    ax.xaxis.set_major_locator(MultipleLocator(250))

    # Zoomed
    zoom = 1.85
    zoom_position = 1
    x1, x2, y1, y2 = 0, 375 \
        , 3e-4, 3e-2

    axins = zoomed_inset_axes(ax, zoom, loc=zoom_position)
    axins.plot(db_h_qe['cnot_count'], db_h_qe['error'], label='IQEB-VQE')
    axins.plot(db_g_qe['cnot_count'], db_g_qe['error'], label='qubit-excitations')
    axins.plot(db_g_efe['cnot_count'], db_g_efe['error'], label='ADAPT-VQE')
    axins.plot(db_g_pwe['cnot_count'], db_g_pwe['error'], label='qubit-ADAPT-VQE')
    axins.fill_between([0, 20000], 1e-9, 1e-3, color='lavender', label='chemical accuracy')
    axins.grid(b=True, which='major', color='black', linestyle='--', linewidth=0.5)
    axins.grid(b=True, which='minor', color='black', linestyle='-.', linewidth=0.3)

    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_yscale('log')
    axins.xaxis.set_minor_locator(MultipleLocator(125))
    axins.xaxis.set_major_locator(MultipleLocator(250))

    mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0", linewidth=.75)

    # ax.legend(loc=9)#, bbox_to_anchor=(1,0.4))

    plt.show()

    print('macaroni')
