import pandas as pd
import matplotlib.pyplot as plt

plt.close('all')

original_df = pd.read_csv('../../../results/iter_vqe_results/'
                          'H4_iqeb_q_exc_r=1.546_22-Feb-2021.csv')
cnot_depth_list = original_df['cnot_depth']

# Device noise
noisy_df = pd.read_csv('../../../results/zhenghao_testing/'
                       'H4_r=1.546_noise_comparison_method=statevector_23-Feb-2021 '
                       '(13:17:34.006525)')
noiseless_E = noisy_df['noiseless E']
noisy_E = noisy_df['noisy E']
time_list = noisy_df['time']

# 1000 shots, no noise
noiseless_qasm_df_0 = pd.read_csv('../../../results/zhenghao_testing/'
                         'H4_r=1.546_noiseless_qasm_method=statevector_25-Feb-2021 '
                         '(22:08:53.934979)')
noiseless_qasm_E = noiseless_qasm_df_0['noisy E']
time_list_0 = noiseless_qasm_df_0['time']

# single qubit gate error p1, two qubit gate error p2, measurement error pm
# 10 shots, p1=0, p2=0.03, pm=0.05
noisy_df_1 = pd.read_csv('../../../results/zhenghao_testing/'
                         'H4_r=1.546_p1=0_p2=0.03_pm=0.05_method=statevector_25-Feb-2021 '
                         '(16:51:35.291605)')
noisy_E_1 = noisy_df_1['noisy E']
time_list_1 = noisy_df_1['time']

# 1000 shots, p1=0, p2=0.03, pm=0.05
noisy_df_2 = pd.read_csv('../../../results/zhenghao_testing/'
                         'H4_r=1.546_p1=0_p2=0.03_pm=0.05_method=statevector_25-Feb-2021 '
                         '(16:55:19.839998)')
noisy_E_2 = noisy_df_2['noisy E']
time_list_2 = noisy_df_2['time']

# 1000 shots, p1=0, p2=0.003, pm=0.005
noisy_df_3 = pd.read_csv('../../../results/zhenghao_testing/'
                         'H4_r=1.546_p1=0_p2=0.003_pm=0.005_method=statevector_25-Feb-2021 '
                         '(21:33:21.736795)')
noisy_E_3 = noisy_df_3['noisy E']

# 1000 shots, p1=0, p2=0.0003, pm=0.0005
noisy_df_4 = pd.read_csv('../../../results/zhenghao_testing/'
                         'H4_r=1.546_p1=0_p2=0.0003_pm=0.0005_method=statevector_25-Feb-2021 '
                         '(21:39:58.763887)')
noisy_E_4 = noisy_df_4['noisy E']

# 1000 shots, p1=0, p2=3e-5, pm=5e-5
noisy_df_5 = pd.read_csv('../../../results/zhenghao_testing/'
                         'H4_r=1.546_p1=0_p2=3e-05_pm=5e-05_method=statevector_25-Feb-2021 '
                         '(21:51:39.621589)')
noisy_E_5 = noisy_df_5['noisy E']

# 10,000 shots, no noise
noiseless_qasm_df_1 = pd.read_csv('../../../results/zhenghao_testing/'
                                  'H4_r=1.546_noiseless_qasm_10000shots_method=statevector_26-Feb-2021 '
                                  '(09:40:42.115244)')
noiseless_qasm_E_1 = noiseless_qasm_df_1['noisy E']

# 1e6 shots, no noise
noiseless_qasm_df_2 = pd.read_csv('../../../results/zhenghao_testing/'
                                  'H4_r=1.546_noiseless_qasm_1000000.0shots_method=automatic_02-Mar-2021 '
                                  '(08:39:46.277502)')
noiseless_qasm_E_2 = noiseless_qasm_df_2['noisy E']


iterations = list(range(1, len(noiseless_E) + 1))

big_list = [noiseless_E,
            noisy_E,
            noisy_E_1, noisy_E_2,
            noisy_E_3,
            noisy_E_5, noiseless_qasm_E,
            noiseless_qasm_E_1, noiseless_qasm_E_2]
legend_list = ['Exact',
               'ibmq_16_melbourne, 10 shots',
               'p_2=3e-2, p_m=5e-2, 10 shots',
               'p_2=3e-2, p_m=5e-2, 1e3 shots',
               'p_2=3e-3, p_m=5e-3, 1e3 shots',
               'p_2=3e-4, p_m=5e-4, 1e3 shots',
               'noiseless QasmBackend, 1e3 shots',
               'noiseless QasmBackend, 1e4 shots',
               'noiseless QasmBackend, 1e6 shots']

plt.figure(1)
for E_list in big_list:
    list_size = len(E_list)
    plt.plot(iterations[0:list_size], E_list)
plt.legend(legend_list)
plt.xlabel('Number of Ansatz Elements')
plt.ylabel('Energy (Hartree)')
plt.title('Noisy comparison for ham_expectation_value for H4(1.546)')

# plt.figure(2)
# plt.plot(iterations[0:len(time_list_0)], time_list_0)
# plt.plot(iterations, time_list)
# plt.plot(iterations[0:len(time_list_1)], time_list_1)
# plt.plot(iterations[0:len(time_list_2)], time_list_2)
# plt.legend(['noiseless 1000 shots, with grouping',
#             'noisy 10 shots, no grouping',
#             'noisy 10 shots, with grouping',
#             'noisy 1000 shots, with grouping'])
# plt.xlabel('Number of Ansatz Elements')
# plt.ylabel('Time (s)')
# plt.title('Run time for ham_expectation_value under noise for H4(1.546)')

plt.figure(3)
for E_list in big_list:
    list_size = len(E_list)
    plt.plot(cnot_depth_list[0:list_size], E_list)
plt.legend(legend_list)
plt.xlabel('CNOT depth')
plt.ylabel('Energy (Hartree)')
plt.title('Noisy comparison for ham_expectation_value for H4(1.546)')

plt.figure(4)
for E_list in big_list[1:len(big_list)]:
    list_size = len(E_list)
    diff_list = E_list - big_list[0][0:list_size]
    plt.plot(cnot_depth_list[0:list_size], diff_list)
plt.legend(legend_list[1:len(legend_list)])
plt.xlabel('CNOT depth')
plt.ylabel('delta E(Hartree)')
plt.yscale('log')
plt.title('Noisy energy difference for ham_expectation_value for H4(1.546')
