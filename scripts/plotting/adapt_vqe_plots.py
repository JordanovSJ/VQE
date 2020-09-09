import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


if __name__ == "__main__":

    db_h_qe = pandas.read_csv('../../results/adapt_vqe_results/vip/BeH2_h_adapt_gsdqe_13-Aug-2020.csv')
    db_g_pwe = pandas.read_csv('../../results/adapt_vqe_results/vip/BeH2_g_adapt_gsdpwe_restricted_combinations_25-Aug-2020.csv')
    db_g_efe = pandas.read_csv('../../results/adapt_vqe_results/vip/BeH2_g_adapt_gsdefe_corrected_09-Sep-2020.csv')
    db_g_qe = pandas.read_csv('../../results/adapt_vqe_results/vip/BeH2_g_adapt_gsdqe_27-Aug-2020.csv')

    fig, ax = plt.subplots()

    ax.plot(db_h_qe['n'], db_h_qe['error'], label='IQEB-VQE')
    ax.plot(db_g_qe['n'], db_g_qe['error'], label='grad-IQEB-VQE')
    ax.plot(db_g_efe['n'], db_g_efe['error'], label='ADAPT-VQE')
    ax.plot(db_g_pwe['n'], db_g_pwe['error'], label='qubit-ADAPT-VQE')
    ax.fill_between([0, 11000], 1e-9, 1e-3, color='lavender')

    ax.set_xlabel('Iterations')
    ax.set_ylabel(r'$E(\theta) - E_{FCI}$, Hartree')
    ax.set_ylim(1e-7, 1e-1)
    ax.set_xlim(0, 250)
    ax.set_yscale('log')

    # Zoomed
    zoom = 2
    zoom_position = 1
    x1, x2, y1, y2 = 0, 40, 4e-4, 3e-2

    axins = zoomed_inset_axes(ax, zoom, loc=zoom_position)
    axins.plot(db_h_qe['n'], db_h_qe['error'], label='IQEB-VQE')
    axins.plot(db_g_qe['n'], db_g_qe['error'], label='qubit-excitations')
    axins.plot(db_g_efe['n'], db_g_efe['error'], label='ADAPT-VQE')
    axins.plot(db_g_pwe['n'], db_g_pwe['error'], label='qubit-ADAPT-VQE')
    axins.fill_between([0, 20000], 1e-9, 1e-3, color='lavender', label='chemical accuracy')

    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_yscale('log')
    # plt.yticks([])
    # plt.xticks([])#[0, x2/3, 2*x2/3, x2],[0,500,1000,1500])

    mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")

    # ax.legend(loc=9)#, bbox_to_anchor=(1,0.4))

    plt.show()

    print('macaroni')
