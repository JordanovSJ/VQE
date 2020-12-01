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

    # db_qe_lih = pandas.read_csv('../../../results/iter_vqe_results/vip/LiH_g_adapt_gsdqe_31-Jul-2020.csv')
    db_qe_lih = pandas.read_csv('../../../results/iter_vqe_results/exc_states/LiH_exc_1_iter_vqe_gsdqe_r=1546_21-Oct-2020.csv')
    db_qe_lih['n_pars'] = numpy.arange(len(db_qe_lih)) + 1
    # db_fe_lih = pandas.read_csv('../../../results/iter_vqe_results/vip/LiH_g_adapt_gsdfe_27-Jul-2020.csv')
    db_fe_lih = pandas.read_csv('../../../results/iter_vqe_results/exc_states/LiH_exc_1_iter_vqe_gsdfe_r=1546_21-Oct-2020.csv')
    db_fe_lih['n_pars'] = numpy.arange(len(db_fe_lih)) + 1

    # db_qe_beh2 = pandas.read_csv('../../../results/iter_vqe_results/vip/BeH2_g_adapt_gsdqe_27-Aug-2020.csv')
    db_qe_beh2 = pandas.read_csv('../../../results/iter_vqe_results/exc_states/BeH2_exc_1_iqeb_n=10_gsdqe_r=275_01-Dec-2020.csv')
    db_qe_beh2['n_pars'] = numpy.arange(len(db_qe_beh2)) + 1
    # db_fe_beh2 = pandas.read_csv('../../../results/iter_vqe_results/vip/BeH2_g_adapt_gsdfe_27-Aug-2020.csv')
    db_fe_beh2 = pandas.read_csv('../../../results/iter_vqe_results/exc_states/BeH2_exc_1_iqeb_n=10_gsdfe_r=275_01-Dec-2020.csv')
    db_fe_beh2['n_pars'] = numpy.arange(len(db_fe_beh2)) + 1

    fig, ax = plt.subplots()

    df_col = 'n_pars'
    linewidth = 0.4
    marker = '_'

    ax.plot(db_qe_lih[df_col], db_qe_lih['error'], label='LiH, qubit excitations', marker=marker, linewidth=linewidth, color='navy')
    ax.plot(db_fe_lih[df_col], db_fe_lih['error'], label='LiH, fermionic excitations', marker=marker, linewidth=linewidth, color='darkred')

    ax.plot(db_qe_beh2[df_col], db_qe_beh2['error'], label='BeH2, qubit excitations', marker=marker, linewidth=linewidth, color='royalblue')
    ax.plot(db_fe_beh2[df_col], db_fe_beh2['error'], label='BeH2, fermionic excitations', marker=marker, linewidth=linewidth, color='indianred')

    ax.fill_between([0, 11000], 1e-15, 1e-3, color='lavender', label='chemical accuracy')

    ax.set_xlabel('Number of ansatz elements/excitations')
    ax.set_ylabel(r'$E(\theta) - E_{FCI}$, Hartree')
    ax.set_ylim(1e-10, 1e-1)
    ax.set_xlim(0, 120)
    ax.set_yscale('log')
    ax.grid(b=True, which='major', color='grey', linestyle='--',linewidth=0.5)

    ax.legend(loc=1)#, bbox_to_anchor=(1,0.4))

    plt.show()

    print('macaroni')
