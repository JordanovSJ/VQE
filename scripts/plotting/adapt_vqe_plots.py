import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas


if __name__ == "__main__":

    db_h_qe = pandas.read_csv('../../results/adapt_vqe_results/vip/LiH_h_adapt_gradient_gsdqe_11-Aug-2020.csv')
    db_g_pwe = pandas.read_csv('../../results/adapt_vqe_results/vip/LiH_g_adapt_gsdpwe_27-Jul-2020.csv')
    db_g_efe = pandas.read_csv('../../results/adapt_vqe_results/vip/LiH_g_adapt_gsdfe_27-Jul-2020.csv')
    db_g_qe = pandas.read_csv('../../results/adapt_vqe_results/vip/LiH_g_adapt_gsdqe_31-Jul-2020.csv')

    plt.plot(db_h_qe['cnot_count'], db_h_qe['error'], label='IQEB-VQE')
    plt.plot(db_g_qe['cnot_count'], db_g_qe['error'], label='qubit-excitations')
    plt.plot(db_g_efe['cnot_count'], db_g_efe['error'], label='ADAPT-VQE')
    plt.plot(db_g_pwe['cnot_count'], db_g_pwe['error'], label='qubit-ADAPT-VQE')
    plt.hlines(1e-3, xmin=0, xmax=1000, linewidth=0.5, label='Chemical accuracy')

    plt.legend()
    plt.xlabel('Number of CNOT gates')
    plt.ylabel(r'$E(\theta) - E_{FCI}$, Hartree')
    # plt.ylim(1e-9, 1e-2)
    plt.xlim(0, 1000)
    plt.yscale('log')

    plt.show()

    print('macaroni')
