import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas
from src.q_systems import *

if __name__ == "__main__":

    db_hf = pandas.DataFrame(columns=['r', 'E'])
    db_hf['r'] = r = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
    db_hf['E'] = [-2.186793421772378, -3.0916972644349183, -3.1355322137028443, -2.966201860939166, -2.7501500435672916,
                  -2.5443021796055967, -2.368421283682794, -2.226815466113642, -2.1167850621241966, -2.033163366704529,
                  -1.9706022457185686]

    db_06 = pandas.read_csv('../../../results/dissociation_curves/H6_iqeb_06.csv')
    fci_rs = db_06['r']
    fci_Es = db_06['fci_E']
    db_hf['error'] = list(numpy.array(db_hf['E']) - numpy.array(fci_Es))
    db_08 = pandas.read_csv('../../../results/dissociation_curves/H6_iqeb_08.csv')
    db_04 = pandas.read_csv('../../../results/dissociation_curves/H6_iqeb_04.csv')

    db_uccsd = pandas.read_csv('../../../results/dissociation_curves/H6_uccsd_31-May-2021.csv')

    plt.plot(db_06['r'], db_06['n_iters'], label=r'IQEB-VQE $\epsilon=10^{-6}$', marker='+', linewidth=0.5, color='green')
    plt.plot(db_08['r'], db_08['n_iters'], label=r'IQEB-VQE $\epsilon=10^{-8}$', marker='+', linewidth=0.5, color='red')
    plt.plot(db_04['r'], db_04['n_iters'], label=r'IQEB-VQE $\epsilon=10^{-8}$', marker='+', linewidth=0.5, color='blue')

    plt.hlines([117] ,xmin=0.5, xmax=3, color='tab:brown', linewidth=1)

    # plt.vlines([1.546], ymax=200, ymin=-100, linewidth=0.75, color='black', label='ground configuration')
    plt.fill_between([0.25, 3.75], 1e-9, 1e-3, color='lavender', label='chemical accuracy')

    plt.xlabel(r'H-H bond distance, $\AA$', fontsize=15)
    # plt.ylabel(r'Number of ansatz parameters')
    plt.ylim(0, 200)
    plt.xlim(0.25, 3.25)
    plt.grid(b=True, which='major', color='grey', linestyle='--', linewidth=0.5)

    plt.show()

    print('macaroni')