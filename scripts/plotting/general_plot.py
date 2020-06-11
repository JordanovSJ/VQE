import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas


if __name__ == "__main__":

    db_data = pandas.read_csv('../../results/dissociation_curves/LiH_dis_curve_10-Jun-2020 (19:17:59.600939).csv')

    # plt.plot(db_data['r'], db_data['error'], label='vqe')
    plt.plot(db_data['r'], db_data['fci_E'], label='fci')
    plt.legend()
    # plt.yscale('log')

    plt.show()

    print('macaroni')
