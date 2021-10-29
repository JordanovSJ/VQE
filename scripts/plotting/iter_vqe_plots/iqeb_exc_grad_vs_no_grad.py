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

    # lih_no_grad_df = pandas.read_csv('../../../results/iter_vqe_results/exc_states/LiH_exc_1_iqeb_q_exc_n=10_r=1546_01-Apr-2021.csv')
    # lih_grad_df = pandas.read_csv('../../../results/iter_vqe_results/exc_states/LiH_exc_1_iqeb_grad_q_exc_n=10_r=1546_02-Apr-2021.csv')

    lih_no_grad_df = pandas.read_csv('../../../results/iter_vqe_results/LiH_exc_0_iqeb_n=1_q_exc_r=1546_28-Jul-2021.csv')
    lih_no_grad_df['i'] = numpy.arange(len(lih_no_grad_df))
    lih_grad_df = pandas.read_csv('../../../results/iter_vqe_results/LiH_iqeb_q_exc_n=1_r=1546_29-Mar-2021.csv')
    lih_grad_df['i'] = numpy.arange(len(lih_grad_df))

    # beh2_no_grad_df = pandas.read_csv('../../../results/iter_vqe_results/exc_states/BeH2_exc_1_iqeb_n=10_vqe_gsdqe_r=1316_25-Nov-2020.csv')
    # beh2_grad_df = pandas.read_csv('../../../results/iter_vqe_results/exc_states/BeH2_exc_1_iqeb_grad_q_exc_n=10_r=1316_02-Apr-2021.csv')

    beh2_no_grad_df = pandas.read_csv('../../../results/iter_vqe_results/BeH2_exc_0_iqeb_n=1_q_exc_r=1316_28-Jul-2021.csv')
    beh2_no_grad_df['i'] = numpy.arange(len(beh2_no_grad_df))
    beh2_grad_df = pandas.read_csv('../../../results/iter_vqe_results/BeH2_iqeb_q_exc_n=1_r=1316_17-Mar-2021.csv')
    beh2_grad_df['i'] = numpy.arange(len(beh2_grad_df))

    fig, ax = plt.subplots()

    # df_col = 'cnot_count'
    df_col = 'i'
    linewidth = 0.4
    marker = '_'

    ax.plot(lih_no_grad_df[df_col], lih_no_grad_df['error'], label='LiH, exc-QEB-ADAPT', marker=marker, linewidth=linewidth, color='blue')
    ax.plot(lih_grad_df[df_col], lih_grad_df['error'], label='LiH, QEB-ADAPT', marker=marker, linewidth=linewidth, color='red')

    ax.plot(beh2_no_grad_df[df_col], beh2_no_grad_df['error'], label='BeH2, exc-QEB-ADAPT', marker=marker, linewidth=linewidth, color='darkblue')
    ax.plot(beh2_grad_df[df_col], beh2_grad_df['error'], label='BeH2, QEB-ADAPT', marker=marker, linewidth=linewidth, color='darkred')

    ax.fill_between([0, 11000], 1e-15, 1e-3, color='lavender', label='chemical accuracy')

    # ax.set_xlabel('Number of CNOTs')
    ax.set_xlabel('Number of qubit evolutions', fontsize=15)
    ax.set_ylabel(r'$E(\theta) - E_{FCI}$, Hartree', fontsize=15)
    # ax.set_ylim(1e-9, 1e-1)
    ax.set_ylim(1e-10, 1e-1)
    # ax.set_xlim(0, 1200)
    ax.set_xlim(0, 150)
    ax.set_yscale('log')
    ax.set_title(r'Ground state energies, $r_{Li-H}=1.546$  $r_{Be-H}=1.316$')
    ax.grid(b=True, which='major', color='grey', linestyle='--',linewidth=0.5)

    ax.legend(loc=1, fontsize=10)#, bbox_to_anchor=(1,0.4))

    plt.show()

    print('macaroni')
