import pandas as pd
import matplotlib.pyplot as plt
from src.molecules.molecules import H4

h4=H4(r=1)
ground_state = h4.fci_energy


# One element, qexc
# no noise, init param 0.1
df_qasm_1 = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_no_noise_shots=1000000.0_18-Mar-2021 (18:42:54.600745).csv')

# p2=1e-6, init param = -0.1, qexc
df_qasm_noise_1 = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=1_tcx=1e-06_0_elements_shots=1000000.0_18-Mar-2021 (23:11:11.885166).csv')

# p2 = 1e-4, init param = -0.1, qexc
df_qasm_noise_4_oneelement = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=1_tcx=0.0001_0_elements_shots=1000000.0_19-Mar-2021 (13:00:48.954616).csv')

# p2 = 1e-4, init param = 0.0001, qexc
df_qasm_noise_4_oneelement_2 = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=0_1_elements_shots=1000000.0_21-Mar-2021 (19:21:37.435522).csv')

# p2 = 1e-6, init param = 1e4, qexc
df_qasm_noise_6_oneelement = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=1_tcx=1e-06_0_elements_shots=1000000.0_19-Mar-2021 (13:03:37.248538).csv')

# qiskit, qexc
df_qiskit_1 = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_qiskitsim_COBYLA_shots=1000000.0_18-Mar-2021 (18:44:22.456696).csv')

# p2 = 1e-6, init param = 1e-4, fexc
df_f_qasm_noise_6_oneelement = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=1e-06_tcx=0_1_elements_shots=1000000.0_21-Mar-2021 (12:07:46.393465).csv')

# p2 = 1e-4, init param = 1e-4, fexc
df_f_qasm_noise_4_oneelement = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=0_1_elements_shots=1000000.0_21-Mar-2021 (17:39:40.980424).csv')

# Five elements
# qiskit, qexc
df_qiskit_5 =pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_qiskitsim_COBYLA_5_elements_shots=1000000.0_19-Mar-2021 (13:16:00.741294).csv')

# p2 = 1e-6, init par = -0.1, qexc
df_qasm_noise_5 = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=1e-06_tcx=0_5_elements_shots=1000000.0_19-Mar-2021 (00:39:02.281904).csv')

# p2 = 1e-6, init par = 1e4, qexc
df_qasm_noise_6_fiveelements = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=5_tcx=1e-06_0_elements_shots=1000000.0_19-Mar-2021 (13:26:58.438456).csv')

# p2 = 1e-4, init par = -0.1, qexc
df_qasm_noise_4_fiveelements = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=5_tcx=0.0001_0_elements_shots=1000000.0_19-Mar-2021 (13:26:02.584855).csv')

# p2 = 1e-4, init par = 0.0001, qexc
df_qasm_noise_4_fiveelements_2 = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=0_5_elements_shots=1000000.0_19-Mar-2021 (16:00:50.233544).csv')

# p2 = 1e-6, init par = 1e-4, fexc
df_f_qasm_noise_6_fiveelements = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=1e-06_tcx=0_5_elements_shots=1000000.0_21-Mar-2021 (12:09:47.758194).csv')

# p2 = 1e-4, init par = 1e-4, fexc
df_f_qasm_noise_4_fiveelements = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=0_5_elements_shots=1000000.0_21-Mar-2021 (17:40:16.619704).csv')

# qexc, p2=1e-4, tcx=10, init par = 1e-4
df_q_qasm_noise_tcx_5elem = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=10_5_elements_shots=1000000.0_22-Mar-2021 (23:25:49.320787).csv')

# fexc, p2=1e-4, tcx=10, init par =1e-4
df_f_qasm_noise_tcx_5elem = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=10_5_elements_shots=1000000.0_23-Mar-2021 (11:34:46.705074).csv')


# plt.figure(1)
# plt.plot(df_qiskit_1['iteration'], df_qiskit_1['energy'], label='QiskitSimBackend')
# plt.plot(df_qasm_1['iteration'], df_qasm_1['energy'], label='QasmBackend, no noise, init_par=0.1')
# plt.plot(df_qasm_noise_1['iteration'], df_qasm_noise_1['energy'], label='QasmBackend, p2=1e-6, init_par=-0.1')
# plt.legend()
# plt.xlabel('Iteration')
# plt.ylabel('Energy (Hartree)')
# plt.title('VQE for one element with COBYLA optimizer')

exact_energy = list(df_qiskit_1['energy'])[-1]
chem_acc = 1e-3

# One element
plt.figure(2)
# plt.plot(df_qasm_1['iteration'], abs(df_qasm_1['energy'] - exact_energy),
#          label='q_exc, shot noise, init_par=0.1')
# plt.plot(df_qasm_noise_1['iteration'], abs(df_qasm_noise_1['energy'] - exact_energy),
#          label='q_exc, p2=1e-6, init_par=-0.1')
# plt.plot(df_qasm_noise_4_oneelement['iteration'], abs(df_qasm_noise_4_oneelement['energy']-exact_energy),
#          label='q_exc, p2=1e-4') # init_par=-0.1
plt.plot(df_qasm_noise_4_oneelement_2['iteration'], abs(df_qasm_noise_4_oneelement_2['energy']-exact_energy),
         label='q_exc, p2=1e-4')#init_par=0.0001
plt.plot(df_f_qasm_noise_4_oneelement['iteration'], abs(df_f_qasm_noise_4_oneelement['energy'] - exact_energy),
         label='f_exc, p2=1e-4')#init_par=0.0001
plt.plot(df_qasm_noise_6_oneelement['iteration'], abs(df_qasm_noise_6_oneelement['energy']-exact_energy),
         label='q_exc, p2=1e-6')#init_par=0.0001

plt.plot(df_f_qasm_noise_6_oneelement['iteration'], abs(df_f_qasm_noise_6_oneelement['energy'] - exact_energy),
         label='f_exc, p2=1e-6')#init_par=0.0001

plt.hlines(chem_acc, 0, max(df_qasm_1['iteration']),
           label='chem accuracy')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Energy accuracy (Hartree)')
plt.yscale('log')
plt.title('VQE for 1 element with COBYLA optimizer')

plt_filename = 'COBYLA_convergence/{}_COBYLA_convergence.png'.format(1)
plt.savefig(plt_filename)

exact_energy_5 = list(df_qiskit_5['energy'])[-1]

# 5 elements
plt.figure(3)
# plt.plot(df_qasm_noise_5['iteration'], abs(df_qasm_noise_5['energy']- exact_energy_5),
#          label='q_exc, p2=1e-6, init_par=-0.1')
plt.plot(df_qasm_noise_4_fiveelements['iteration'], abs(df_qasm_noise_4_fiveelements['energy']-exact_energy_5),
         label='q_exc, p2=1e-4') #init_par=-0.1
# plt.plot(df_qasm_noise_4_fiveelements_2['iteration'], abs(df_qasm_noise_4_fiveelements_2['energy']-exact_energy_5),
#          label='q_exc, p2=1e-4') #init_par=0.0001
plt.plot(df_f_qasm_noise_4_fiveelements['iteration'], abs(df_f_qasm_noise_4_fiveelements['energy']-exact_energy_5),
         label='f_exc, p2=1e-4') #init_par=0.0001

plt.plot(df_qasm_noise_6_fiveelements['iteration'], abs(df_qasm_noise_6_fiveelements['energy']-exact_energy_5),
         label='q_exc, p2=1e-6') #init_par=0.0001

plt.plot(df_f_qasm_noise_6_fiveelements['iteration'], abs(df_f_qasm_noise_6_fiveelements['energy']-exact_energy_5),
         label='f_exc, p2=1e-6') #init_par=0.0001
#
# plt.plot(df_q_qasm_noise_tcx_5elem['iteration'], abs(df_q_qasm_noise_tcx_5elem['energy']-exact_energy_5),
#          label='q_exc, p2=1e-4, tcx=10') #init_par=0.0001
# plt.plot(df_f_qasm_noise_tcx_5elem['iteration'], abs(df_f_qasm_noise_tcx_5elem['energy']-exact_energy_5),
#          label='f_exc, p2=1e-4, tcx=10') #init_par=0.0001
plt.hlines(chem_acc, 0, max(df_qasm_noise_5['iteration']),
           label='chem accuracy')
# plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Energy accuracy (Hartree)')
plt.yscale('log')
plt.title('VQE for 5 elements with COBYLA optimizer')

plt_filename = 'COBYLA_convergence/{}_COBYLA_convergence.png'.format(5)
plt.savefig(plt_filename)