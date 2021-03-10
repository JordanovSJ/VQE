import matplotlib.pyplot as plt
import pandas as pd
from src.q_systems import *

plt.close('all')

molecule = H2()

data = pd.read_csv('../csv_folder/H2_different_nshots_14-Feb-2021 (16:20:43.774108).csv')

plt.figure(1)
plt.plot(data['n_shots'], data['expectation_value'], 'rx')
plt.plot(data['n_shots'], data['ref_result'])
plt.xscale('log')
plt.xlabel('n_shots')
plt.ylabel('Expectation value/Hartree')
plt.legend('QasmBackend', 'QiskitSimBackend')
plt.title('Expectation value against n_shots values for {}'.format(molecule.name))

plt.figure(2)
plt.plot(data['n_shots'], data['time'])
plt.xscale('log')
plt.xlabel('n_shots')
plt.ylabel('time/s')
plt.title('Time for different n_shots for {}'.format(molecule.name))