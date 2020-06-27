import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas


if __name__ == "__main__":

    db_e_qe = pandas.read_csv('../../results/adapt_vqe_results/LiH_energy_adapt_SDQE_05-Jun-2020.csv')
    db_e_efe = pandas.read_csv('../../results/adapt_vqe_results/LiH_energy_adapt_SDEFE_05-Jun-2020.csv')
    db_g_efe = pandas.read_csv('../../results/adapt_vqe_results/LiH_grad_adapt_EFE_16-Jun-2020.csv')
    db_g_pwe = pandas.read_csv('../../results/adapt_vqe_results/LiH_grad_adapt_pwe_21-Jun-2020.csv')

    plt.plot(db_e_qe['n'], db_e_qe['error'], label='energy_based_qe')
    plt.plot(db_e_efe['n'], db_e_efe['error'], label='energy_based_efe')
    plt.plot(db_g_efe['n'], db_g_efe['error'], label='grad_based_efe')
    plt.plot(db_g_pwe['n'], db_g_pwe['error'], label='grad_based_pwe')

    plt.legend()
    plt.yscale('log')

    plt.show()

    print('macaroni')
