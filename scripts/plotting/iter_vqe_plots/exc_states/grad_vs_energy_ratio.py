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


    db_n_1_bh2 = pandas.read_csv('../../../../results/iter_vqe_results/exc_states/BeH2_exc_1_iqeb_q_exc_n=1_r=1316_11-Jun-2021.csv')
    db_n_5_bh2 = pandas.read_csv('../../../../results/iter_vqe_results/exc_states/BeH2_exc_1_iqeb_q_exc_n=5_r=1316_09-Jun-2021.csv')
    db_n_10_bh2 = pandas.read_csv('../../../../results/iter_vqe_results/exc_states/BeH2_exc_1_iqeb_n=10_vqe_gsdqe_r=1316_25-Nov-2020.csv')
    db_n_20_bh2 = pandas.read_csv('../../../../results/iter_vqe_results/exc_states/BeH2_exc_1_iqeb_vqe_n=20_gsdqe_r=1316_26-Nov-2020.csv')

    db_n_1_lih = pandas.read_csv('../../../../results/iter_vqe_results/exc_states/LiH_exc_1_iqeb_q_exc_n=1_r=1546_11-Jun-2021.csv')
    db_n_5_lih = pandas.read_csv('../../../../results/iter_vqe_results/exc_states/LiH_exc_1_iqeb_q_exc_n=5_r=1546_09-Jun-2021.csv')
    db_n_10_lih = pandas.read_csv('../../../../results/iter_vqe_results/exc_states/LiH_exc_1_iqeb_q_exc_n=10_r=1546_01-Apr-2021.csv')
    db_n_20_lih = pandas.read_csv('../../../../results/iter_vqe_results/exc_states/LiH_exc_1_iqeb_q_exc_n=20_r=1546_09-Jun-2021.csv')

    df_col = 'cnot_count'

    dEs = numpy.logspace(-9, 0, 50)
    interp_bh2_1 = numpy.interp(dEs, numpy.flip(db_n_1_bh2['error'].values), numpy.flip(db_n_1_bh2[df_col].values))
    interp_bh2_5 = numpy.interp(dEs, numpy.flip(db_n_5_bh2['error'].values), numpy.flip(db_n_5_bh2[df_col].values))
    interp_bh2_10 = numpy.interp(dEs, numpy.flip(db_n_10_bh2['error'].values), numpy.flip(db_n_10_bh2[df_col].values))
    interp_bh2_30 = numpy.interp(dEs, numpy.flip(db_n_20_bh2['error'].values), numpy.flip(db_n_20_bh2[df_col].values))

    interp_lih_1 = numpy.interp(dEs, numpy.flip(db_n_1_lih['error'].values), numpy.flip(db_n_1_lih[df_col].values))
    interp_lih_5 = numpy.interp(dEs, numpy.flip(db_n_5_lih['error'].values), numpy.flip(db_n_5_lih[df_col].values))
    interp_lih_10 = numpy.interp(dEs, numpy.flip(db_n_10_lih['error'].values), numpy.flip(db_n_10_lih[df_col].values))
    interp_lih_30 = numpy.interp(dEs, numpy.flip(db_n_20_lih['error'].values), numpy.flip(db_n_20_lih[df_col].values))

    fig, ax = plt.subplots()

    linewidth = 0.4
    marker = '_'

    print(sum(1 - interp_lih_5 / interp_lih_1)/len(1 - interp_lih_5 / interp_lih_1))
    print(sum(1 - interp_lih_10 / interp_lih_1)/len(1 - interp_lih_10 / interp_lih_1))
    print(sum(1 - interp_lih_30 / interp_lih_1)/len(1 - interp_lih_30 / interp_lih_1))

    print(sum(1 - interp_bh2_5 / interp_bh2_1) / len(1 - interp_bh2_5 / interp_bh2_1))
    print(sum(1 - interp_bh2_10 / interp_bh2_1) / len(1 - interp_bh2_10 / interp_bh2_1))
    print(sum(1 - interp_bh2_30 / interp_bh2_1) / len(1 - interp_bh2_30 / interp_bh2_1))

    print(max(1 - interp_lih_5 / interp_lih_1))
    print(max(1 - interp_lih_10 / interp_lih_1))
    print(max(1 - interp_lih_30 / interp_lih_1))

    print(max(1 - interp_bh2_5 / interp_bh2_1))
    print(max(1 - interp_bh2_10 / interp_bh2_1))
    print(max(1 - interp_bh2_30 / interp_bh2_1))


    ax.plot(dEs, 1 - interp_bh2_10/interp_bh2_1, label=r'BeH$_2$, n=10', linewidth=linewidth, color='dodgerblue')
    ax.plot(dEs, 1 - interp_bh2_30/interp_bh2_1, label=r'BeH$_2$, n=30', linewidth=linewidth, color='tomato')

    ax.plot(dEs, 1 - interp_lih_10 / interp_lih_1, label=r'LiH$_2$, n=10', linewidth=linewidth, color='blue')
    ax.plot(dEs, 1 - interp_lih_30 / interp_lih_1, label=r'LiH$_2$, n=30', linewidth=linewidth, color='red')

    ax.set_ylabel('Reduction of CNOTs, %')
    ax.set_xlabel(r'$E(\theta) - E_{FCI}$, Hartree')
    ax.set_xscale('log')

    ax.legend(loc=1)#, bbox_to_anchor=(1,0.4))

    plt.show()

    print('macaroni')
