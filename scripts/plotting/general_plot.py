import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas


if __name__ == "__main__":

    db_data = pandas.read_csv('../../results/dissociation_curves/HF_dis_curve_04-Jun-2020 (14:49:22.755434).csv')

    plt.plot(db_data['r'], db_data['error'], label='vqe')
    # plt.plot(db_data['r'], db_data['fci_E'], label='fci')
    plt.legend()
    # plt.yscale('log')

    plt.show()

    print('macaroni')
