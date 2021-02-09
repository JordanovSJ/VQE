import pandas

import sys
sys.path.append('../../')

import matplotlib.pyplot as plt
import numpy as np
from src.vqe_runner import VQERunner
from src.q_systems import *
from src.ansatz_element_sets import *
from src.backends import QiskitSimBackend
from src.utils import *
from src.iter_vqe_utils import *
from src.cache import *

from scripts.zhenghao.iqeb_fun_li import iqeb_litest

r_num = 20
r_start = 0.5
r_end = 2.0
r_list = np.linspace(r_start, r_end, r_num)

fci_energy =np.zeros(r_num)
vqe_energy =np.zeros(r_num)
hf_energy = np.zeros(r_num)
i=0

molecule_type = H2

for r in r_list:
    molecule = molecule_type(r)
    fci_energy[i] = molecule.fci_energy
    hf_energy[i] = molecule.hf_energy
    vqe_result = iqeb_litest(r, molecule)
    vqe_energy[i] = vqe_result.fun
    i = i +1


plt.figure(1)
plt.plot(r_list, fci_energy)
plt.plot(r_list, hf_energy)
plt.plot(r_list, vqe_energy, 'rx')
plt.title('Dissociation curve for {}'.format(molecule.name))
plt.legend(['fci', 'hf', 'iqeb vqe'])
plt.xlabel('H-H bond distances, Angstroms')
plt.ylabel('Energy, Hartree')
plt.show()

plt.figure(2)
plt.plot(r_list, fci_energy)
plt.title('Dissociation curve for {}'.format(molecule.name))
plt.legend(['fci'])
plt.xlabel('r')
plt.ylabel('energy')
plt.show()

# plt.figure(2)
# plt.plot(r_list, list(fci_energy_0))
# plt.show()

# time_stamp =datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")
# data = {'r': r_list, 'fci': fci_energy, 'hf': hf_energy, 'iqeb': vqe_energy}
# df = pandas.DataFrame(data)
#df.to_csv('{}_dissociation_curve_{}'.format(molecule.name, time_stamp))
