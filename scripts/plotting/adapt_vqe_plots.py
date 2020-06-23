import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas


if __name__ == "__main__":

    db_efe = pandas.read_csv('../../results/adapt_vqe_results/LiH_grad_adapt_EFE_16-Jun-2020.csv')
    db_qe = pandas.read_csv('../../results/adapt_vqe_results/LiH_grad_adapt_pwe_21-Jun-2020.csv')

    plt.plot(db_qe['n'], db_qe['error'], label='qubit_excitations')
    plt.plot(db_efe['n'], db_efe['error'], label='fermi_excitations')
    plt.legend()
    plt.yscale('log')

    plt.show()

    print('macaroni')
