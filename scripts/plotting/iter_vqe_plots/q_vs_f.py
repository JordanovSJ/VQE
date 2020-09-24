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

    db_qe_lih = pandas.read_csv('../../../results/adapt_vqe_results/vip/LiH_g_adapt_gsdqe_31-Jul-2020.csv')
    db_fe_lih = pandas.read_csv('../../../results/adapt_vqe_results/vip/LiH_g_adapt_gsdfe_27-Jul-2020.csv')

    db_qe_beh2 = pandas.read_csv('../../../results/adapt_vqe_results/vip/BeH2_g_adapt_gsdqe_27-Aug-2020.csv')
    db_fe_beh2 = pandas.read_csv('../../../results/adapt_vqe_results/vip/BeH2_g_adapt_gsdfe_27-Aug-2020.csv')

    fig, ax = plt.subplots()

    df_col = 'n'
    linewidth = 0.4
    marker = '_'

    ax.plot(db_qe_lih[df_col], db_qe_lih['error'], label='LiH, qubit excitations', marker=marker, linewidth=linewidth, color='peru')
    ax.plot(db_fe_lih[df_col], db_fe_lih['error'], label='LiH, fermionic excitations', marker=marker, linewidth=linewidth, color='blue')

    ax.plot(db_qe_beh2[df_col], db_qe_beh2['error'], label='BeH2, qubit excitations', marker=marker, linewidth=linewidth, color='orange')
    ax.plot(db_fe_beh2[df_col], db_fe_beh2['error'], label='BeH2, fermionic excitations', marker=marker, linewidth=linewidth, color='cornflowerblue')

    ax.fill_between([0, 11000], 1e-15, 1e-3, color='lavender', label='chemical accuracy')

    ax.set_xlabel('Number of iterations/parameters')
    ax.set_ylabel(r'$E(\theta) - E_{FCI}$, Hartree')
    ax.set_ylim(1e-9, 1e-1)
    ax.set_xlim(0, 120)
    ax.set_yscale('log')
    ax.grid(b=True, which='major', color='grey', linestyle='--',linewidth=0.5)

    ax.legend(loc=1)#, bbox_to_anchor=(1,0.4))

    plt.show()

    print('macaroni')
