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

    db_g_iqeb = pandas.read_csv('../../../results/adapt_vqe_results/vip/LiH_g_adapt_gsdqe_31-Jul-2020.csv')
    db_adapt_pairs = pandas.read_csv('../../../results/adapt_vqe_results/vip/LiH_g_adapt_gsdfe_27-Jul-2020.csv')

    fig, ax = plt.subplots()

    df_col = 'n'
    linewidth = 0.4
    marker = '_'

    ax.plot(db_g_iqeb[df_col], db_g_iqeb['error'], label='qubit excitations', marker=marker, linewidth=linewidth)
    ax.plot(db_adapt_pairs[df_col], db_adapt_pairs['error'], label='fermionic excitations', marker=marker, linewidth=linewidth)
    ax.fill_between([0, 11000], 1e-15, 1e-3, color='lavender', label='chemical accuracy')

    ax.set_xlabel('Number of iterations/parameters')
    ax.set_ylabel(r'$E(\theta) - E_{FCI}$, Hartree')
    ax.set_ylim(1e-9, 1e-2)
    ax.set_xlim(0, 70)
    ax.set_yscale('log')
    ax.grid(b=True, which='major', color='grey', linestyle='--',linewidth=0.5)

    ax.legend(loc=1)#, bbox_to_anchor=(1,0.4))

    plt.show()

    print('macaroni')
