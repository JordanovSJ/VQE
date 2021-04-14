import pandas as pd
import matplotlib.pyplot as plt
from src.molecules.molecules import H4

molecule = H4(r=1)
ground_state_E = molecule.fci_energy
chem_acc = 1e-3

# qexc, p2=1e-6, 11 inspired pars and rest 0.01
df_q_6 = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=1e-06_tcx=0_15_elements_shots=1000000.0_22-Mar-2021 (00:12:59.384017).csv')

# fexc, p2=1e-6, 11 inspired pars and rest 0.01
df_f_6 = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=1e-06_tcx=0_15_elements_shots=1000000.0_22-Mar-2021 (09:32:43.987574).csv')

# qexc, p2=1e-4, 11 inspired pars and rest 0.01
df_q_4 = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=0_15_elements_shots=1000000.0_22-Mar-2021 (23:00:52.350533).csv')

# fexc, p2=1e-5, 11 inspired pars and rest 0.01
df_f_4 = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=0_15_elements_shots=1000000.0_23-Mar-2021 (21:13:58.220540).csv')

plt.figure(1)
plt.plot(df_q_4['iteration'], abs(df_q_4['energy']-ground_state_E),
         label='qexc, p2=1e-4')
plt.plot(df_f_4['iteration'], abs(df_f_4['energy']-ground_state_E),
         label='fexc, p2=1e-4')
plt.plot(df_q_6['iteration'], abs(df_q_6['energy']-ground_state_E),
         label='qexc, p2=1e-6')
plt.plot(df_f_6['iteration'], abs(df_f_6['energy']-ground_state_E),
         label='fexc, p2=1e-6')

plt.hlines(chem_acc, 0, max(df_q_6['iteration']),
           label='chem accuracy')
plt.yscale('log')
# plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Energy accuracy (Hartree)')
plt.title('VQE for 15 elements with COBYLA optimizer')

plt_filename = 'COBYLA_convergence/{}_COBYLA_convergence.png'.format(15)
plt.savefig(plt_filename)