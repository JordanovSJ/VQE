import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas


if __name__ == "__main__":
    db_g_fe = pandas.read_csv('../../results/iter_vqe_results/vip/LiH_g_adapt_gsdfe_27-Jul-2020.csv')
    db_g_spin_fe = pandas.read_csv(
        '../../results/iter_vqe_results/vip/LiH_g_adapt_gsdfe_spin_complement_27-Aug-2020.csv')

    fig, ax = plt.subplots()

    ax.plot(db_g_fe['cnot_count'], db_g_fe['error'], label='individual excitations')
    ax.plot(db_g_spin_fe['cnot_count'], db_g_spin_fe['error'], label='spin-complement excitations')
    ax.fill_between([0, 11000], 1e-9, 1e-3, color='lavender')

    ax.set_xlabel('Iterations')
    ax.set_ylabel(r'$E(\theta) - E_{FCI}$, Hartree')
    # ax.set_ylim(1e-7, 1e-1)
    # ax.set_xlim(0, 250)
    ax.set_yscale('log')

    plt.legend()

    plt.show()

    print('macaroni')
