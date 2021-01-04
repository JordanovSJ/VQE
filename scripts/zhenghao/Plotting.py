import matplotlib.pyplot as plt
import pandas as pd
from src.q_systems import *

plt.close('all')

molecule = H2()

data = pd.read_csv('{}_dissociation_curve.csv'.format(molecule.name))

data_hf_fci_dif = abs(data['hf']-data['fci'])
data_iqeb_fci_dif = abs(data['iqeb']-data['fci'])

plt.figure(1)
plt.plot(data['r'], data['fci'])
plt.plot(data['r'], data['hf'])
plt.plot(data['r'], data['iqeb'], 'rx')
plt.xlabel('H-H bond distance, Angstroms')
plt.ylabel('Energy, Hartree')
plt.legend(['fci','hf','iqeb'])
plt.title('Dissociation curves for {}'.format(molecule.name))
plt.ylim([-1.2, -0.7])
plt.xlim([0.4, 2.1])

plt.figure(2)
plt.plot(data['r'], data_hf_fci_dif)
plt.plot(data['r'], data_iqeb_fci_dif)
plt.xlabel('H-H bond distance, Angstroms')
plt.ylabel('Energy - FCI energy, Hartree')
plt.legend(['hf','iqeb'])
plt.title('Dissociation curve difference for {}'.format(molecule.name))
plt.yscale("log")
plt.ylim([10**-17, 10**0])
plt.xlim([0.4, 2.1])



