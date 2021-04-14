from src.molecules.molecules import H2
import matplotlib.pyplot as plt
import numpy as np

num_data = 20
fci_energy = np.zeros(num_data)
r_list = np.zeros(num_data)

for idx in range(num_data):
    r = 0.1*idx + 0.1
    molecule = H2(r=r)
    fci_energy[idx] = molecule.fci_energy
    r_list[idx] = r

plt.plot(r_list, fci_energy)
plt.show()