import pandas as pd
import matplotlib.pyplot as plt
from src.molecules.molecules import H4

molecule = H4(r=1)
ground_state = molecule.fci_energy

# start with all 0.0001
df_1 = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_qiskitsim_COBYLA_15_elements_shots=1000000.0_19-Mar-2021 (18:13:36.720892).csv')

# start with all 0.01
df_2 = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_qiskitsim_COBYLA_15_elements_shots=1000000.0_19-Mar-2021 (18:23:22.205300).csv')

# start with var pars from 11, and rest 0.0001
df_3 = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_qiskitsim_COBYLA_15_elements_shots=1000000.0_21-Mar-2021 (12:53:40.550111).csv')

# start with var pars from 11, and rest 0.
df_4 = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_qiskitsim_COBYLA_15_elements_shots=1000000.0_21-Mar-2021 (12:57:51.681627).csv')

# start with var pars from 11, and rest 0.01
df_6 = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_qiskitsim_COBYLA_15_elements_shots=1000000.0_21-Mar-2021 (23:54:03.302743).csv')

# BFGS, all 0.01
df_5 = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_qiskitsim_BFGS_15_elements_shots=1000000.0_19-Mar-2021 (18:44:59.264480).csv')

# 19 elements, COBYLA, inspired 15, rest 0.01
df_7 = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_qiskitsim_COBYLA_19_elements_shots=1000000.0_22-Mar-2021 (23:08:09.749466).csv')

# plt.figure(1)
# plt.plot(df_1['iteration'], abs(df_1['energy']-ground_state), label='all 0.0001')
# plt.plot(df_2['iteration'], abs(df_2['energy']-ground_state), label='all 0.01')
# # plt.plot(df_3['iteration'], df_3['energy'], label='inspired 11, rest 0.0001')
# plt.plot(df_4['iteration'], abs(df_4['energy']-ground_state), label='inspired 11, rest 0.')
# plt.plot(df_6['iteration'], abs(df_6['energy']-ground_state), label='inpired 11, rest 0.01')
# plt.plot(df_5['iteration'], abs(df_5['energy']-ground_state), label='BFGS, all 0.01')
# plt.legend()
# plt.yscale('log')
# plt.xlabel('Iteration')
# plt.ylabel('Energy (Hartree)')
# plt.title('15 q ansatz elements, qiskitsimbackend, COBYLA')

plt.figure(2)
plt.plot(df_7['iteration'], abs(df_7['energy']-ground_state), label='inspired 15, rest 0.01')
plt.legend()
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Energy (Hartree)')
plt.title('19 q ansatz elements, qiskitsimbackend, COBYLA')