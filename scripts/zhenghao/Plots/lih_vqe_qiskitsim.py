import pandas as pd
import matplotlib.pyplot as plt
from src.molecules.molecules import LiH
import numpy as np

df_dict = {}

df_dict[1] = pd.read_csv('../../../results/zhenghao_testing/LiH_vqe_q_exc_qiskitsim_COBYLA_1_elements_shots=1000000.0_31-Mar-2021 (12:16:15.271887).csv')

df_dict[3] = pd.read_csv('../../../results/zhenghao_testing/LiH_vqe_q_exc_qiskitsim_COBYLA_3_elements_shots=1000000.0_31-Mar-2021 (12:16:58.029394).csv')

df_dict[5] = pd.read_csv('../../../results/zhenghao_testing/LiH_vqe_q_exc_qiskitsim_COBYLA_5_elements_shots=1000000.0_31-Mar-2021 (12:04:05.898268).csv')

df_dict[8] = pd.read_csv('../../../results/zhenghao_testing/LiH_vqe_q_exc_qiskitsim_COBYLA_8_elements_shots=1000000.0_31-Mar-2021 (12:18:30.521941).csv')

df_dict[11] = pd.read_csv('../../../results/zhenghao_testing/LiH_vqe_q_exc_qiskitsim_COBYLA_11_elements_shots=1000000.0_31-Mar-2021 (12:03:18.294272).csv')

df_dict[15] = pd.read_csv('../../../results/zhenghao_testing/LiH_vqe_q_exc_qiskitsim_COBYLA_15_elements_shots=1000000.0_31-Mar-2021 (12:23:43.115187).csv')

chem_acc = 1e-3

molecule = LiH(r=1.25)
ground_state = molecule.fci_energy

plt.figure(1)
for num_elem in df_dict.keys():
    df_toplot = df_dict[num_elem]
    plt.plot(df_toplot['iteration'], (df_toplot['energy']-ground_state),
             label = '{} elements'.format(num_elem))
plt.hlines(chem_acc, 0, max(df_dict[15]['iteration']),
           label='chem accuracy')
plt.legend()
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Energy difference from true ground state')
plt.title('Running vqe for first n elements for Q Ansatz for LiH with COBYLA')


df_iqeb_f_exc = pd.read_csv('../../../results/iter_vqe_results/LiH_iqeb_eff_f_exc_r=1.25_31-Mar-2021.csv')
df_iqeb_q_exc = pd.read_csv('../../../results/iter_vqe_results/LiH_iqeb_q_exc_r=1.25_19-Nov-2020.csv')

plt.figure(2)
plt.plot(np.arange(1, len(df_iqeb_q_exc['E'])+1, 1), df_iqeb_q_exc['E']-ground_state, label='q exc')
plt.plot(np.arange(1, len(df_iqeb_f_exc['E'])+1, 1), df_iqeb_f_exc['E']-ground_state, label='f exc')
plt.hlines(chem_acc, 0, max(len(df_iqeb_q_exc['E'])+1, len(df_iqeb_f_exc['E'])+1),
           label='chem accuracy')
plt.legend()
plt.xlabel('Num element')
plt.ylabel('Energy difference from true ground state')
plt.title('Running iqeb vqe for LiH')

plt.figure(3)
plt.plot(np.arange(1, len(df_iqeb_q_exc['E'])+1, 1), df_iqeb_q_exc['cnot_count'],
         label='q exc')
plt.plot(np.arange(1, len(df_iqeb_f_exc['E'])+1, 1), df_iqeb_f_exc['cnot_count'],
         label='f exc')
plt.legend()
plt.xlabel('Num element')
plt.ylabel('CNOT count')
plt.title('CNOT count from LiH iqeb')

plt.figure(4)
plt.plot(np.arange(1, len(df_iqeb_q_exc['E'])+1, 1),
         [df_iqeb_f_exc['cnot_count'][idx] - df_iqeb_q_exc['cnot_count'][idx]
          for idx in range(len(df_iqeb_q_exc['cnot_count']))])
plt.xlabel('Num element')
plt.ylabel('CNOT count difference')
plt.title('CNOT count difference from LiH iqeb')

