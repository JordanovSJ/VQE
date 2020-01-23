import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

if __name__ == "__main__":

    fixed_ansatz = pd.read_csv('results/fix_ansatz_h2.csv')
    uccsd_ansatz = pd.read_csv('results/uccsd_ansatz_h2.csv')

    fix_results = []
    uccsd_results = []

    r = []
    E_fix = []
    E_uccsd = []

    for element in fixed_ansatz.to_numpy():
        data = ast.literal_eval(element[1])
        fix_results.append(data)
        r.append(data['r'])
        E_fix.append(data['E'])

    for element in uccsd_ansatz.to_numpy():
        data = ast.literal_eval(element[1])
        uccsd_results.append(data)
        E_uccsd.append(data['E'])

    r = np.array(r)
    E_uccsd = np.array(E_uccsd)
    E_fix = np.array(E_fix)

    plt.plot(r, E_fix + 1, '.')
    plt.plot(r, E_uccsd + 1, '.')

    print(min(E_fix))
    print(min(E_uccsd))

    plt.show()



    print(fixed_ansatz)