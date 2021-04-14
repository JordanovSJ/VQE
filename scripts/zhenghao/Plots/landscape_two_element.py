import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

# prob_1 = 0.01, prob_2=0.001, prob_meas=0.001 time_single_gate=0, time_cx=900, time_meas=0, t1=50000.0, t2=50000.0
# df = pd.read_csv('../../../results/zhenghao_testing/h4_landscape_qiskit_sim_backend.csv')

# prob_2= prob_meas = 1e-6
df_lessnoise = pd.read_csv('../../../results/zhenghao_testing/h4_landscape_p2=1e-6_qiskit_sim_backend.csv')

# prob_1 = 0.001, prob_2=0.01, prob_meas=0.01 time_single_gate=0, time_cx=900, time_meas=0, t1=50000.0, t2=50000.0
df_noise_2 = pd.read_csv('../../../results/zhenghao_testing/h4_landscape_qiskit_sim_backend_28-Mar-2021 (10:47:08.749735).csv')

# prob_1 = 0, prob_2=0.0001, prob_meas=0.0001 time_single_gate=0, time_cx=10, time_meas=0, t1=50000.0, t2=50000.0
df_noise_3 = pd.read_csv('../../../results/zhenghao_testing/h4_landscape_qiskit_sim_backend_28-Mar-2021 (10:49:22.871989).csv')

df = df_noise_3

pars_1 = np.array(df['pars_1'])
pars_2 = np.array(df['pars_2'])
qiskitsim_energy = np.array(df['qiskitsim_energy'])
noisy_energy = np.array(df['noisy_energy'])

pars_1_lessnoise_full = np.array(df_lessnoise['pars_1'])
pars_2_lessnoise_full = np.array(df_lessnoise['pars_2'])
lessnoisy_energy_full = np.array(df_lessnoise['noisy_energy'])

pars_1_lessnoise = []
pars_2_lessnoise = []
lessnoisy_energy = []

for idx in range(len(pars_1_lessnoise_full)):
    if pars_2_lessnoise_full[idx] <=0.01:
        pars_1_lessnoise.append(pars_1_lessnoise_full[idx])
        pars_2_lessnoise.append(pars_2_lessnoise_full[idx])
        lessnoisy_energy.append(lessnoisy_energy_full[idx])

fig_1 = plt.figure(1)
ax = fig_1.add_subplot(111, projection='3d')
surf = ax.plot_trisurf(pars_1, pars_2, qiskitsim_energy, cmap=cm.coolwarm)
# fig_1.colorbar(surf)
ax.set_xlabel('theta 1')
ax.set_ylabel('theta 2')
ax.set_zlabel('Energy')
ax.set_xlim([-0.25, 0])
ax.set_ylim([-0.1, 0])
fig_1.colorbar(surf, shrink=0.5, aspect=5)
plt.title('QiskitSimBackend')

fig_2 = plt.figure(2)
ax_1 = fig_2.add_subplot(111, projection='3d')
surf_1 = ax_1.plot_trisurf(pars_1, pars_2, noisy_energy, cmap=cm.coolwarm)
ax_1.set_xlabel('theta 1')
ax_1.set_ylabel('theta 2')
ax_1.set_zlabel('Energy')
ax.set_xlim([-0.25, 0])
ax.set_ylim([-0.1, 0])
fig_2.colorbar(surf_1, shrink=0.5, aspect=5)
plt.title('p_1=0, p_2=1e-4, p_meas=1e-4, t_cx=10')

fig_3 = plt.figure(3)
ax_2 = fig_3.add_subplot(111, projection='3d')
surf_2 = ax_2.plot_trisurf(pars_1_lessnoise, pars_2_lessnoise, lessnoisy_energy,
                           cmap=cm.coolwarm)
ax_2.set_xlabel('theta 1')
ax_2.set_ylabel('theta 2')
ax_2.set_zlabel('Energy')
# ax.set_xlim([-0.25, 0])
ax.set_ylim([-0.1, 0])
fig_3.colorbar(surf_2, shrink=0.5, aspect=5)
plt.title('p_1=0, p_2=1e-6, p_meas=1e-6, t_cx=0')



# fig_3 = plt.figure(3)
# ax_2 = fig_3.add_subplot(111, projection='3d')
# surf_2 = ax_2.plot_trisurf(pars_1, pars_2, noisy_energy-qiskitsim_energy)
# ax_2.set_xlabel('theta 1')
# ax_2.set_ylabel('theta 2')
# ax_2.set_zlabel('Energy difference')
# plt.title('Landscape difference between QiskitSimBackend and QasmBackend p2=1e-6')

plt.show()