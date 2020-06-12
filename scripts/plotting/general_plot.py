import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas


if __name__ == "__main__":

    db_data = pandas.read_csv('../../results/dissociation_curves/LiH_adapt_SDEFE_dis_curve_10-Jun-2020.csv')

    # plt.plot(db_data['r'], db_data['error'], label='vqe', marker='*')
    plt.plot(db_data['r'], db_data['E'], label='E', marker='*')
    # plt.plot(db_data['r'], db_data['E'], label='fci')
    # plt.plot(db_data['r'], db_data['fci_E'], label='fci', marker='*')
    plt.legend()
    # plt.yscale('log')

    plt.show()

    print('macaroni')
