from src.q_system import *
from src.molecules.molecules import *

import numpy
import pandas


if __name__ == "__main__":

    exc_state = 9

    r_max = 3
    r_min = 0.75
    n_rs = 20
    rs = numpy.arange(n_rs+1)*(r_max-r_min)/n_rs + r_min

    fci_Es = []

    for r in rs:
        print(r)
        molecule = BeH2(r=r)
        fci_Es.append(molecule.calculate_energy_eigenvalues(exc_state+1)[exc_state])

    df = pandas.DataFrame(columns=['r', 'fci_E'])
    df['r'] = rs
    df['fci_E'] = fci_Es

    print(fci_Es)

    df.to_csv('BeH2_exc_{}_fci_E.csv'.format(exc_state))

    print('pasta')
