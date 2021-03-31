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
    # db_qe_lih = pandas.read_csv('../../../results/iter_vqe_results/LiH_iqeb_q_exc_n=1_r=1546_29-Mar-2021.csv')
    db_qe_lih = pandas.read_csv('../../../results/iter_vqe_results/LiH_iqeb_q_exc_no_comps_n=1_r=1546_18-Mar-2021.csv')
    db_qe_lih['n_pars'] = numpy.arange(len(db_qe_lih)) + 1
    # db_qe_lih_r3 = pandas.read_csv('../../../results/iter_vqe_results/LiH_iqeb_n=1_gsdqe_r=3_03-Dec-2020.csv')
    db_qe_lih_r3 = pandas.read_csv('../../../results/iter_vqe_results/LiH_iqeb_q_exc_no_comps_n=1_r=3_18-Mar-2021.csv')
    db_qe_lih_r3['n_pars'] = numpy.arange(len(db_qe_lih_r3)) + 1

    # db_fe_lih_r3 = pandas.read_csv('../../../results/iter_vqe_results/LiH_iqeb_eff_f_exc_r=3_15-Mar-2021.csv')
    db_fe_lih_r3 = pandas.read_csv('../../../results/iter_vqe_results/LiH_iqeb_eff_f_exc_no_comps_n=1_r=3_29-Mar-2021.csv')
    db_fe_lih_r3['n_pars'] = numpy.arange(len(db_fe_lih_r3)) + 1
    # db_fe_lih = pandas.read_csv('../../../results/iter_vqe_results/LiH_iqeb_eff_f_exc_r=1546_15-Mar-2021.csv')
    db_fe_lih = pandas.read_csv('../../../results/iter_vqe_results/LiH_iqeb_eff_f_exc_no_comps_n=1_r=1546_29-Mar-2021.csv')
    db_fe_lih['n_pars'] = numpy.arange(len(db_fe_lih)) + 1

    # db_qe_beh2 = pandas.read_csv('../../../results/iter_vqe_results/BeH2_iqeb_q_exc_n=1_r=1316_17-Mar-2021.csv')
    db_qe_beh2 = pandas.read_csv('../../../results/iter_vqe_results/BeH2_iqeb_q_exc_no_comps_n=1_r=1316_18-Mar-2021.csv')
    db_qe_beh2['n_pars'] = numpy.arange(len(db_qe_beh2)) + 1
    # db_fe_beh2 = pandas.read_csv('../../../results/iter_vqe_results/BeH2_iqeb_eff_f_exc_n=1_r=1316_22-Mar-2021.csv')
    db_fe_beh2 = pandas.read_csv('../../../results/iter_vqe_results/BeH2_iqeb_eff_f_exc_no_comps_n=1_r=1316_29-Mar-2021.csv')
    db_fe_beh2['n_pars'] = numpy.arange(len(db_fe_beh2)) + 1

    # db_qe_beh2_r3 = pandas.read_csv('../../../results/iter_vqe_results/BeH2_iqeb_n=1_gsdqe_r=3_05-Dec-2020.csv')
    db_qe_beh2_r3 = pandas.read_csv('../../../results/iter_vqe_results/BeH2_iqeb_q_exc_no_comps_n=1_r=3_29-Mar-2021.csv')
    db_qe_beh2_r3['n_pars'] = numpy.arange(len(db_qe_beh2_r3)) + 1
    # db_fe_beh2_r3 = pandas.read_csv('../../../results/iter_vqe_results/BeH2_iqeb_eff_f_exc_n=1_r=3_22-Mar-2021.csv')
    db_fe_beh2_r3 = pandas.read_csv('../../../results/iter_vqe_results/BeH2_iqeb_eff_f_exc_no_comps_n=1_r=3_29-Mar-2021.csv')
    db_fe_beh2_r3['n_pars'] = numpy.arange(len(db_fe_beh2_r3)) + 1

    fig, ax = plt.subplots()

    df_col = 'n'
    linewidth = 0.4
    marker = '_'

    # ax.plot(db_qe_lih[df_col], db_qe_lih['error'], label=r'Qubit evolutions, $r_{Li-H}=1.316 \AA$', marker=marker, linewidth=linewidth, color='blue')
    # ax.plot(db_fe_lih[df_col], db_fe_lih['error'], label='Fermionic evolutions, $r_{Li-H}=1.316 \AA$', marker=marker, linewidth=linewidth, color='red')
    #
    # ax.plot(db_qe_lih_r3[df_col], db_qe_lih_r3['error'], label='Qubit evolutions, $r_{Li-H}=3 \AA$', marker=marker, linewidth=linewidth, color='darkblue')
    # ax.plot(db_fe_lih_r3[df_col], db_fe_lih_r3['error'], label='Fermionic evolutions, $r_{Li-H}=3 \AA$', marker=marker, linewidth=linewidth, color='darkred')

    ax.plot(db_qe_beh2[df_col], db_qe_beh2['error'], label='Qubit evolutions, $r_{Be-H}=1.546 \AA$', marker=marker, linewidth=linewidth, color='blue')
    ax.plot(db_fe_beh2[df_col], db_fe_beh2['error'], label='Fermionic evolutions, $r_{Be-H}=1.546 \AA$', marker=marker, linewidth=linewidth, color='red')

    ax.plot(db_qe_beh2_r3[df_col], db_qe_beh2_r3['error'], label='Qubit evolutions, $r_{Be-H}=3 \AA$', marker=marker, linewidth=linewidth, color='darkblue')
    ax.plot(db_fe_beh2_r3[df_col], db_fe_beh2_r3['error'], label='Fermionic evolutions, $r_{Be-H}=3 \AA$', marker=marker, linewidth=linewidth, color='darkred')

    ax.fill_between([0, 11000], 1e-15, 1e-3, color='lavender', label='chemical accuracy')

    ax.set_xlabel('Number of ansatz elements/evolutions')
    ax.set_ylabel(r'$E(\theta) - E_{FCI}$, Hartree')
    ax.set_ylim(1e-9, 1e-0)
    ax.set_xlim(0, 140)
    ax.set_yscale('log')
    ax.grid(b=True, which='major', color='grey', linestyle='--',linewidth=0.5)

    ax.legend(loc=1)#, bbox_to_anchor=(1,0.4))

    plt.show()

    print('macaroni')
