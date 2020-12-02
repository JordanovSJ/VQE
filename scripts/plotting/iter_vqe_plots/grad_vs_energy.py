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

    db_n_1 = pandas.read_csv('../../../results/iter_vqe_results/exc_states/BeH2_exc_1_iter_vqe_gsdqe_r=1316_24-Nov-2020.csv')
    db_n_10 = pandas.read_csv('../../../results/iter_vqe_results/exc_states/BeH2_exc_1_iqeb_n=10_vqe_gsdqe_r=1316_25-Nov-2020.csv')
    db_n_20 = pandas.read_csv('../../../results/iter_vqe_results/exc_states/BeH2_exc_1_iqeb_vqe_n=20_gsdqe_r=1316_26-Nov-2020.csv')

    fig, ax = plt.subplots()

    df_col = 'cnot_count'
    linewidth = 0.4
    marker = '_'

    ax.plot(db_n_1[df_col], db_n_1['error'], label='n=1', marker=marker, linewidth=linewidth, color='forestgreen')
    ax.plot(db_n_10[df_col], db_n_10['error'], label='n=10', marker=marker, linewidth=linewidth, color='orange')
    ax.plot(db_n_20[df_col], db_n_20['error'], label='n=20', marker=marker, linewidth=linewidth, color='cornflowerblue')
    ax.fill_between([0, 11000], 1e-15, 1e-3, color='lavender', label='chemical accuracy')

    ax.set_xlabel('Number of CNOTs')
    ax.set_ylabel(r'$E(\theta) - E_{FCI}$, Hartree')
    ax.set_ylim(1e-8, 1e-1)
    ax.set_xlim(0, 1200)
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

    ax.legend(loc=1)#, bbox_to_anchor=(1,0.4))

    plt.show()

    print('macaroni')
