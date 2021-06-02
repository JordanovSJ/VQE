import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas
from src.q_system import *

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
    db_uccsd = pandas.read_csv('../../../results/dissociation_curves/H6_uccsd_31-May-2021.csv')

    plt.plot(db_hf['r'], db_hf['E'], label='HF', marker='+', linewidth=0.5, color='orange')
    plt.plot(db_uccsd['r'], db_uccsd['E'], label=r'UCCSD', marker='*', linewidth=0.5, color='blue')
    plt.plot(db_06['r'], db_06['E'], label=r'IQEB-VQE $\epsilon=10^{-6}$ Hartree', marker='*', linewidth=0.5, color='green')
    plt.plot(db_08['r'], db_08['E'], label=r'IQEB-VQE $\epsilon=10^{-8}$ Hartree', marker='*', linewidth=0.5, color='red')
    plt.plot(fci_rs, fci_Es, label='FCI energy', linewidth=0.5, color='purple')
    # plt.vlines([1.546], ymax=200, ymin=-100, linewidth=0.75, color='black', label='equilibrium configuration')
    plt.fill_between([0.5, 3.75], 1e-9, 1e-3, color='lavender', label='chemical accuracy')

    # plt.xlabel(r'Li-H bond distance, $\AA$')
    # plt.ylabel('Energy, Hartree')
    plt.ylim(-3.25, -1.95)
    plt.xlim(0.25, 3.25)
    # plt.legend(loc=2, fontsize=8)
    plt.grid(b=True, which='major', color='grey', linestyle='--', linewidth=0.5)

    plt.show()

    print('macaroni')