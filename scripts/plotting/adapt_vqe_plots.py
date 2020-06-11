import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas


if __name__ == "__main__":

    db_efe = pandas.read_csv('../../results/adapt_vqe_results/LiH_SDEFE_05-Jun-2020.csv')
    db_qe = pandas.read_csv('../../results/adapt_vqe_results/LiH_SDQE_05-Jun-2020.csv')

    plt.plot(db_qe['n'], np.log(db_qe['error'])/np.log(10), label='qubit_excitations')
    plt.plot(db_efe['n'], np.log(db_efe['error'])/np.log(10), label='fermi_excitations')
    plt.legend()
    # plt.yscale('log')

    plt.show()

    print('macaroni')
