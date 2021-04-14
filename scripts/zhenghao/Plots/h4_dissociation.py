from src.molecules.molecules import H4
import numpy as np
import matplotlib.pyplot as plt

r_range = np.arange(0.5, 5, 0.1)

energy_dict = {}

for r in r_range:
    molecule = H4(r=r)
    energy_dict[r]=molecule.fci_energy

plt.figure(1)
plt.plot(energy_dict.keys(), energy_dict.values())
