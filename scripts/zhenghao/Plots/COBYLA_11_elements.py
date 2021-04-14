import pandas as pd
import matplotlib.pyplot as plt
from src.molecules.molecules import H4

molecule = H4(r=1)
ground_state_E = molecule.fci_energy

# qasm, f_exc, no noise, init par = -0.1
df_f_nonoise = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_no_noise_shots=1000000.0_18-Mar-2021 (19:04:31.495815).csv')

# qasm, f_exc, no noise, init par = 0.0001
df_f_nonoise_2 = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_no_noise_shots=1000000.0_21-Mar-2021 (12:12:46.322952).csv')

# qasm, q_exc no noise, init par = -0.1
df_q_nonoise = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_no_noise_shots=1000000.0_18-Mar-2021 (19:01:31.912931).csv')

# qasm, qexc, p2=1e-6, init par = -0.1
df_q_noise = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=1e-06_tcx=0_shots=1000000.0_18-Mar-2021 (19:12:16.969257).csv')

# qasm, fexc, p2=1e-6, init par = -0.1
df_f_noise = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=1e-06_tcx=0_shots=1000000.0_18-Mar-2021 (19:13:52.870310).csv')

# qasm, qexc, p2=1e-4, init par = 0.0001
df_q_noise_4 = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=0_11_elements_shots=1000000.0_21-Mar-2021 (22:53:09.328452).csv')

# qasm, fexc, p2=1e-4, init par = 0.0001
df_f_noise_4 = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=0_11_elements_shots=1000000.0_22-Mar-2021 (22:47:33.008108).csv')

# qexc, p2=1e-4, tcx=10, init par = 0.0001
df_q_noise_4_tcx = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=10_11_elements_shots=1000000.0_24-Mar-2021 (10:37:17.543387).csv')

# fexc, p2=1e-4, tcx=10, init par = 0.0001
df_f_noise_4_tcx = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=10_11_elements_shots=1000000.0_24-Mar-2021 (10:38:55.731061).csv')

# qiskit, qexc, init par = -0.1
df_q_qiskit = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_qiskitsim_COBYLA_shots=1000000.0_18-Mar-2021 (18:34:42.717855).csv')

# qiskit, fexc, init par = -0.1
df_f_qiskit = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_qiskitsim_COBYLA_11_elements_shots=1000000.0_19-Mar-2021 (10:30:11.507600).csv')



chem_acc = 1e-3

plt.figure(1)
# plt.plot(df_q_nonoise['iteration'], abs(df_q_nonoise['energy']-ground_state_E),
#          label='q_exc, shot noise, init par=-0.1')
# plt.plot(df_f_nonoise['iteration'], abs(df_f_nonoise['energy']-ground_state_E),
#          label='f_exc, shot noise, init par=-0.1')
# plt.plot(df_f_nonoise_2['iteration'], abs(df_f_nonoise_2['energy']-ground_state_E),
#          label='f_exc, shot noise, init par=0.0001')

plt.plot(df_q_noise_4['iteration'], abs(df_q_noise_4['energy']-ground_state_E),
         label='q_exc, p2=1e-4, init par=0.0001')
plt.plot(df_f_noise_4['iteration'], abs(df_f_noise_4['energy']-ground_state_E),
         label='f_exc, p2=1e-4, init par=0.0001')

plt.plot(df_q_noise['iteration'], abs(df_q_noise['energy']-ground_state_E),
         label='q_exc, p2=1e-6, init par=-0.1')
plt.plot(df_f_noise['iteration'], abs(df_f_noise['energy']-ground_state_E),
         label='f_exc, p2=1e-6, , init par=-0.1')

# plt.plot(df_q_noise_4_tcx['iteration'], abs(df_q_noise_4_tcx['energy']-ground_state_E),
#          label='q_exc, p2=1e-4, tcx=10, init par=0.0001')
# plt.plot(df_f_noise_4_tcx['iteration'], abs(df_f_noise_4_tcx['energy']-ground_state_E),
#          label='f_exc, p2=1e-4, tcx=10, init par=0.0001')
# plt.plot(df_q_qiskit['iteration'], abs(df_q_qiskit['energy']-ground_state_E),
#          label='q_exc, exact')
# plt.plot(df_f_qiskit['iteration'], abs(df_f_qiskit['energy']-ground_state_E),
#          label='f_exc, exact')
plt.hlines(chem_acc, 0, max(df_f_noise['iteration']),
           label='chem accuracy')
# plt.legend()
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Energy accuracy (Hartree)')
plt.title('VQE for 11 elements with COBYLA optimizer')

plt_filename = 'COBYLA_convergence/{}_COBYLA_convergence.png'.format(11)
plt.savefig(plt_filename)

#
# param_str_dict = {
#     # 'param_q_nonoise' : list(df_q_nonoise['params'])[-1],
#     'q exact' : list(df_q_qiskit['params'])[-1],
#     # 'param_q_noise' : list(df_q_noise['params'])[-1],
#     'f shot noise, init par=-0.1' : list(df_f_nonoise['params'])[-1],
#     'f shot noise, init par=0.0001': list(df_f_nonoise_2['params'])[-1],
#     'f exact' : list(df_f_qiskit['params'])[-1],
#     # 'param_f_noise' : list(df_f_noise['params'])[-1]
#
# }

param_dict = {}
#
# for param_name in param_str_dict.keys():
#     param_str = param_str_dict[param_name]
#
#     param_str_copy = param_str.replace('[', '').replace(']', '').replace('\n', '')
#     param_str_list = param_str_copy.split(' ')
#     while '' in param_str_list:
#         param_str_list.remove('')
#
#     param_list = [float(str_i) for str_i in param_str_list]
#
#     param_dict[param_name] = param_list
#
#     print('{}: {}'.format(param_name, param_list))
# #
# def param_dist(param_list_1, param_list_2):
#     assert len(param_list_1) == len(param_list_2)
#
#     list_len = len(param_list_1)
#     distance_2 = 0
#     for i in range(list_len):
#         distance_2 += (param_list_1[i] - param_list_2[i])**2
#
#     return distance_2
#
# param_q_qiskit = param_dict['q exact']
# param_f_qiskit = param_dict['f exact']

# for param_name in param_dict.keys():
#
#     param = param_dict[param_name]
#
#     if 'f ' in param_name:
#         distance = param_dist(param, param_f_qiskit)
#     else:
#         distance = param_dist(param, param_q_qiskit)
#
#     print('Distance squared is {} for {}'.format(distance, param_name))
