import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

if __name__ == "__main__":

    fixed_ansatz = pd.read_csv('results/fix_ansatz_h2.csv')
    uccsd_ansatz = pd.read_csv('results/uccsd_ansatz_h2.csv')

    fixed_ansatz_2 = pd.read_csv('results/fix_ansatz_h2_2.csv')
    uccsd_ansatz_2 = pd.read_csv('results/uccsd_ansatz_h2_2.csv')

    fix_results = []
    uccsd_results = []

    r = []
    E_fix = []
    E_uccsd = []

    for element in list(fixed_ansatz.to_numpy())+list(fixed_ansatz_2.to_numpy()):
        data = ast.literal_eval(element[1])
        fix_results.append(data)
        r.append(data['r'])
        E_fix.append(data['E'])

    for element in list(uccsd_ansatz.to_numpy())+list(uccsd_ansatz_2.to_numpy()):
        data = ast.literal_eval(element[1])
        uccsd_results.append(data)
        E_uccsd.append(data['E'])

    r = np.array(r)
    E_uccsd = np.array(E_uccsd)
    E_fix = np.array(E_fix)

    plt.plot(r, E_fix, 'x',  markersize=5, color='r', label='Hardware efficient ansatz')
    plt.plot(r, E_uccsd, 'o', markersize=1.5, color='b', label='UCCSD ansatz')
    plt.ylabel('Energy [Hartree]')
    plt.xlabel(r'Nuclei separation [$\AA$]')
    plt.title(r'$H_2$ energy as function of nuclei separation')
    plt.ylim(-1.15, -0.9)
    plt.vlines(0.741, -1.1, -1.15, linewidth=0.5)
    plt.legend()

    print(min(E_fix))
    print(min(E_uccsd))

    plt.show()



    print(fixed_ansatz)