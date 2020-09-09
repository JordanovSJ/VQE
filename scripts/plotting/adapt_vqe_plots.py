import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas


if __name__ == "__main__":

    db_h_qe = pandas.read_csv('../../results/adapt_vqe_results/vip/BeH2_h_adapt_gsdqe_13-Aug-2020.csv')
    db_g_pwe = pandas.read_csv('../../results/adapt_vqe_results/vip/BeH2_g_adapt_gsdpwe_restricted_combinations_25-Aug-2020.csv')
    db_g_efe = pandas.read_csv('../../results/adapt_vqe_results/vip/BeH2_g_adapt_gsdfe_27-Aug-2020.csv')
    db_g_qe = pandas.read_csv('../../results/adapt_vqe_results/vip/BeH2_g_adapt_gsdqe_27-Aug-2020.csv')

    plt.plot(db_h_qe['cnot_count'], db_h_qe['error'], label='IQEB-VQE')
    plt.plot(db_g_qe['cnot_count'], db_g_qe['error'], label='qubit-excitations')
    plt.plot(db_g_efe['cnot_count'], db_g_efe['error'], label='ADAPT-VQE')
    plt.plot(db_g_pwe['cnot_count'], db_g_pwe['error'], label='qubit-ADAPT-VQE')
    plt.hlines(1e-3, xmin=0, xmax=11000, linewidth=0.5)

    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel(r'$E(\theta) - E_{FCI}$, Hartree')
    plt.ylim(1e-7, 1e-1)
    plt.xlim(0, 11000)
    plt.yscale('log')

    plt.show()

    print('macaroni')
