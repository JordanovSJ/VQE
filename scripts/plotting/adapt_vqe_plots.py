import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas


if __name__ == "__main__":

    db_h_qe = pandas.read_csv('../../results/adapt_vqe_results/vip/LiH_h_adapt_gsdqe_27-Jul-2020.csv')
    db_g_pwe = pandas.read_csv('../../results/adapt_vqe_results/vip/LiH_g_adapt_gsdpwe_27-Jul-2020.csv')
    db_g_efe = pandas.read_csv('../../results/adapt_vqe_results/vip/LiH_g_adapt_gsdfe_27-Jul-2020.csv')
    db_g_qe = pandas.read_csv('../../results/adapt_vqe_results/vip/LiH_g_adapt_gsdqe_31-Jul-2020.csv')

    plt.plot(db_h_qe['cnot_count'], db_h_qe['error'], label='h_qe')
    plt.plot(db_g_qe['cnot_count'], db_g_qe['error'], label='g_qe')
    plt.plot(db_g_efe['cnot_count'], db_g_efe['error'], label='g_efe')
    plt.plot(db_g_pwe['cnot_count'], db_g_pwe['error'], label='g_pwe')

    plt.legend()
    plt.yscale('log')

    plt.show()

    print('macaroni')
