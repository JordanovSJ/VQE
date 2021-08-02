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
    lih_exc_1_qe = pandas.read_csv('../../../../results/iter_vqe_results/exc_states/LiH_exc_1_iqeb_q_exc_n=10_r=1546_01-Apr-2021.csv')
    lih_exc_1_qe['n_pars'] = numpy.arange(len(lih_exc_1_qe)) + 1
    lih_exc_1_fe = pandas.read_csv('../../../../results/iter_vqe_results/exc_states/LiH_exc_1_iqeb_n=10_eff_f_exc_r=1546_06-Apr-2021.csv')
    lih_exc_1_fe['n_pars'] = numpy.arange(len(lih_exc_1_fe)) + 1

    beh2_exc_1_qe = pandas.read_csv('../../../../results/iter_vqe_results/exc_states/BeH2_exc_1_iqeb_n=10_vqe_gsdqe_r=1316_25-Nov-2020.csv')
    beh2_exc_1_qe['n_pars'] = numpy.arange(len(beh2_exc_1_qe)) + 1
    beh2_exc_1_fe = pandas.read_csv('../../../../results/iter_vqe_results/exc_states/BeH2_exc_1_iqeb_vqe_n=10_gsdfe_r=1316_01-Dec-2020.csv')
    beh2_exc_1_fe['n_pars'] = numpy.arange(len(beh2_exc_1_fe)) + 1

    beh2_exc_1_qe_r3 = pandas.read_csv('../../../../results/iter_vqe_results/exc_states/BeH2_exc_1_iqeb_n=10_gsdqe_r=3_30-Nov-2020.csv')
    beh2_exc_1_qe_r3['n_pars'] = numpy.arange(len(beh2_exc_1_qe_r3)) + 1
    beh2_exc_1_fe_r3 = pandas.read_csv('../../../../results/iter_vqe_results/exc_states/BeH2_exc_1_iqeb_n=10_gsdfe_r=3_30-Nov-2020.csv')
    beh2_exc_1_fe_r3['n_pars'] = numpy.arange(len(beh2_exc_1_fe_r3)) + 1

    fig, ax = plt.subplots()

    df_col = 'n'
    linewidth = 0.4
    marker = '_'

    # ax.plot(lih_exc_1_qe[df_col], lih_exc_1_qe['error'], label=r'Qubit exc', marker=marker, linewidth=linewidth, color='blue')
    # ax.plot(lih_exc_1_fe[df_col], lih_exc_1_fe['error'], label=r'Fermionic exc.', marker=marker, linewidth=linewidth, color='red', alpha=0.6)

    ax.plot(beh2_exc_1_qe[df_col], beh2_exc_1_qe['error'], label=r'BeH$_2$, qubit excitations', marker=marker, linewidth=linewidth, color='blue')
    ax.plot(beh2_exc_1_fe[df_col], beh2_exc_1_fe['error'], label=r'BeH$_2$, fermionic excitations', marker=marker, linewidth=linewidth, color='red', alpha=0.6)

    # ax.plot(beh2_exc_1_qe_r3[df_col], beh2_exc_1_qe_r3['error'], label=r'BeH$_2$, qubit exc.', marker=marker, linewidth=linewidth, color='red')
    # ax.plot(beh2_exc_1_fe_r3[df_col], beh2_exc_1_fe_r3['error'], label=r'BeH$_2$, fermionic exc.', marker=marker, linewidth=linewidth, color='blue', alpha=0.6)

    ax.fill_between([0, 11000], 1e-15, 1e-3, color='lavender', label='chemical accuracy')

    ax.set_xlabel('Number of qubit/fermionic excitation evolutions', fontsize=15)
    ax.set_ylabel(r'$E(\theta) - E_{FCI}$, Hartree', fontsize=15)
    ax.set_ylim(1e-10, 1e-1)
    ax.set_xlim(0, 100)
    # ax.set_xlim(0, 35)

    ax.set_yscale('log')
    ax.grid(b=True, which='major', color='grey', linestyle='--',linewidth=0.5)

    # ax.legend(loc=1, fontsize=15)#, bbox_to_anchor=(1,0.4))

    plt.show()

    print('macaroni')
