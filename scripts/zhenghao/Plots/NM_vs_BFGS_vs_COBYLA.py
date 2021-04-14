import pandas as pd
import matplotlib.pyplot as plt
from src.molecules.molecules import H4
from mpl_toolkits.mplot3d import Axes3D

h4= H4(r=1)

# Not adaptive 11 ansatz elements
df1 = pd.read_csv('../../../results/zhenghao_testing/'
                  'H4_vqe_q_exc_qiskitsim_Nelder-Mead_shots=1000000.0_17-Mar-2021 (15:00:05.317493).csv')
# Adaptive 11 ansatz elements
df2 = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_qiskitsim_Nelder-Mead_shots=1000000.0_17-Mar-2021 (15:16:57.598759).csv')

# Adaptive 5 elements
df3 = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_qiskitsim_Nelder-Mead_shots=1000000.0_17-Mar-2021 (18:36:20.033557).csv')
# Not adaptive 5 elements
df4 = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_qiskitsim_Nelder-Mead_shots=1000000.0_17-Mar-2021 (18:37:25.707605).csv')

# Not adaptive 3 elements
df5 = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_qiskitsim_Nelder-Mead_shots=1000000.0_17-Mar-2021 (18:42:41.869462).csv')

params_qiskit_3_elements = df5['params']
x_qiskit_3 = []
y_qiskit_3 = []
z_qiskit_3 = []
for element in params_qiskit_3_elements:
    element_1 = element.replace('[', '')
    element_2 = element_1.replace(']', '')
    split_list = element_2.split(' ')
    while '' in split_list:
        split_list.remove('')
    assert len(split_list) == 3
    x_qiskit_3.append(float(split_list[0]))
    y_qiskit_3.append(float(split_list[1]))
    z_qiskit_3.append(float(split_list[2]))
assert len(params_qiskit_3_elements) == len(x_qiskit_3)

# Adaptive 3 elements
df6 = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_qiskitsim_Nelder-Mead_shots=1000000.0_17-Mar-2021 (18:43:06.135924).csv')

# f exc Nelder Mead no noise 11 ansatz elements
df_f_eleven = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_Nelder-Mead_no_noise_shots=1000000.0_17-Mar-2021 (16:09:31.099573).csv')

# f exc Nelder Mead no noise 3 ansatz elements
df_f_three = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_Nelder-Mead_no_noise_shots=1000000.0_17-Mar-2021 (20:07:21.890420).csv')


# q exc Nelder Mead no noise 11 ansatz elements
df_q_eleven = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_Nelder-Mead_no_noise_shots=1000000.0_17-Mar-2021 (16:08:33.677710).csv')

# q exc Nelder Mead no noise 3 ansatz elements
df_q_three = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_Nelder-Mead_no_noise_shots=1000000.0_17-Mar-2021 (20:06:04.394335).csv')

params_qasm_3_elements = df_q_three['params']
x_qasm_3 = []
y_qasm_3 = []
z_qasm_3 = []
for element in params_qasm_3_elements:
    element_1 = element.replace('[', '')
    element_2 = element_1.replace(']', '')
    split_list = element_2.split(' ')
    while '' in split_list:
        split_list.remove('')
    assert len(split_list) == 3
    x_qasm_3.append(float(split_list[0]))
    y_qasm_3.append(float(split_list[1]))
    z_qasm_3.append(float(split_list[2]))
assert len(params_qasm_3_elements) == len(x_qasm_3)

# q exc Nelder Mead no noise 1 ansatz element
df_q_one = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_Nelder-Mead_no_noise_shots=1000000.0_17-Mar-2021 (22:49:16.149408).csv')

# q exc qiskitsim Nelder Mead 1 ansatz element
df_q_qiskit_one = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_qiskitsim_Nelder-Mead_shots=1000000.0_17-Mar-2021 (22:43:47.708394).csv')

# bfgs
df_bfgs = pd.read_csv(
    '../../../results/zhenghao_testing/vqe_bfgs/H4_vqe_q_exc_qiskitsim_BFGS_shots=1000000.0_17-Mar-2021 (15:53:01.338045).csv')

# cobyla
df_cobyla = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_qiskitsim_COBYLA_shots=1000000.0_18-Mar-2021 (18:34:42.717855).csv')

ground_state = h4.fci_energy
#
# plt.figure(1)
# plt.plot(df1['iteration'], df1['energy'], label='NM, 11 elements')
# plt.plot(df2['iteration'], df2['energy'], label='a-NM, 11 elements')
#
# plt.plot(df4['iteration'], df4['energy'], label='NM, 5 elements')
# plt.plot(df3['iteration'], df3['energy'], label='a-NM, 5 elements')
#
# plt.plot(df5['iteration'], df5['energy'], label='NM, 3 elements')
# plt.plot(df6['iteration'], df6['energy'], label='a-NM, 3 elements')
#
#
# plt.plot(df_bfgs['iteration'], df_bfgs['energy'], label='BFGS, 11 elements')
#
# plt.hlines(ground_state, 0, max(df1['iteration']), label='fci energy')
#
# plt.legend()
# plt.xlabel('Iteration')
# plt.ylabel('Energy (Hartree)')
# plt.title('VQE run for comparing optimizers')
#
plt.figure(2)
plt.hlines(1e-3, 0, max(df1['iteration']), label='Chem accuracy')
plt.plot(df1['iteration'], abs(df1['energy']-ground_state), label='NM, 11 elements')
plt.plot(df2['iteration'], abs(df2['energy']-ground_state), label='a-NM, 11 elements')
plt.plot(df_bfgs['iteration'], abs(df_bfgs['energy']-ground_state), label='BFGS, 11 elements')
plt.plot(df_cobyla['iteration'], abs(df_cobyla['energy']-ground_state), label='COBYLA, 11 elements')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Energy difference (Hartree)')
plt.yscale('log')
plt.title('VQE run for comparing optimizers')

# plt.figure(3)
# plt.plot(df2['iteration'], df2['energy'], label='QiskitSimBackend')
# plt.plot(df_q_eleven['iteration'], df_q_eleven['energy'], label='QasmBackend no noise')
# plt.hlines(ground_state, 0, max(df2['iteration']), label='fci energy')
# plt.legend()
# plt.xlabel('Iteration')
# plt.ylabel('Energy')
# plt.title('VQE run with adaptive Nelder Mead for 11 q_exc elements')
#
# plt.figure(4)
# plt.plot(df5['iteration'], df5['energy'], label='QiskitSimBackend')
# plt.plot(df_q_three['iteration'], df_q_three['energy'], label='QasmBackend no noise')
# plt.legend()
# plt.xlabel('Iteration')
# plt.ylabel('Energy')
# plt.title('VQE run with Nelder Mead for 3 q_exc elements')
#
# fig_5 = plt.figure(5)
# ax = fig_5.add_subplot(111, projection='3d')
# ax.plot(x_qiskit_3, y_qiskit_3, z_qiskit_3, label='QiskitSimBackend')
# ax.scatter(x_qiskit_3[-1], y_qiskit_3[-1], z_qiskit_3[-1], c='r', label='QiskitSimBackend end point')
# ax.legend()
# ax.set_xlabel('theta_1')
# ax.set_xlim3d(-0.25, 0)
# ax.set_ylabel('theta_2')
# ax.set_ylim3d(-0.15, 0)
# ax.set_zlabel('theta_3')
# ax.set_zlim3d(-0.15, 0.05)
# plt.title('VQE run with Nelder Mead for 3 q_exc elements, parameters')
# plt.show()
#
# fig_6 = plt.figure(6)
# ax1 = fig_6.add_subplot(111, projection='3d')
# ax1.plot(x_qasm_3, y_qasm_3, z_qasm_3, label='QasmBackend no noise')
# ax1.scatter(x_qasm_3[-1], y_qasm_3[-1], z_qasm_3[-1], c='r', label='QasmBackend end point')
# ax1.legend()
# ax1.set_xlabel('theta_1')
# ax1.set_xlim3d(-0.25, 0)
# ax1.set_ylabel('theta_2')
# ax1.set_ylim3d(-0.15, 0)
# ax1.set_zlabel('theta_3')
# ax1.set_zlim3d(-0.15, 0.05)
# plt.title('VQE run with Nelder Mead for 3 q_exc elements, parameters')
# plt.show()