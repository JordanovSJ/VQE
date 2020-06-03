import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas


if __name__ == "__main__":

    db_qe = pandas.read_csv('../../results/adapt_vqe_results/HF_02-Jun-2020_adapt_SDQE.csv')
    db_efe = pandas.read_csv('../../results/adapt_vqe_results/HF_02-Jun-2020_adapt_SDEFE.csv')

    plt.plot(db_qe['cnot_count'], db_qe['error'], label='qubit_excitations')
    plt.plot(db_efe['cnot_count'], db_efe['error'], label='fermi_excitations')
    plt.legend()
    plt.yscale('log')

    plt.show()

    print('macaroni')
