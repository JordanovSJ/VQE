import pandas as pd
import matplotlib.pyplot as plt

plt.close('all')

df_1 = pd.read_csv('../../../results/zhenghao_testing/'
                   'H4_r=1.546_iqeb_ansatz_compare_method=statevector_04-Mar-2021 '
                   '(16:13:05.603622).csv')
df_2 = pd.read_csv('../../../results/zhenghao_testing/'
                   'H4_r=1.546_iqeb_ansatz_compare_method=statevector_04-Mar-2021 '
                   '(17:35:33.726776).csv')
ref_E = list(df_1['ref E']) + list(df_2['ref E'])
noiseless_E = list(df_1['noiseless E']) + list(df_2['noiseless E'])
noiseless_time = list(df_1['noiseless time']) + list(df_2['noiseless time'])
n_shots = list(df_1['n_shots'])+list(df_2['n_shots'])
noisy_E = list(df_1['noisy E'])+list(df_2['noisy E'])
noisy_time = list(df_1['noisy time']) + list(df_2['noisy time'])

plt.figure(1)
plt.plot(n_shots, ref_E)
plt.plot(n_shots, noiseless_E, 'x')
plt.plot(n_shots, noisy_E, 'x')
plt.xlabel('Shot number')
plt.xscale('log')
plt.ylabel('Ham expectation value (Hartree)')
plt.legend(['Exact', 'Noiseless QasmSimulator', 'Ibmq melbourne noise'])
plt.title('Ham expectation value vs number of shots')

plt.figure(2)
plt.plot(n_shots, [abs(x-y) for x,y in zip(noiseless_E, ref_E)], 'x')
plt.plot(n_shots, [abs(x-y) for x,y in zip(noisy_E, ref_E)], 'x')
plt.legend(['Noiseless QasmSimulator', 'IBMQ-Melbourne noise'])
plt.xlabel('Shot number')
plt.xscale('log')
plt.ylabel('delta E (Hartree)')
plt.yscale('log')
plt.title('Accuracy vs shot number')

plt.figure(3)
plt.plot(n_shots, noiseless_time)
plt.plot(n_shots, noisy_time)
plt.xlabel('Shot number')
plt.xscale('log')
plt.ylabel('Time')
plt.legend(['Noiseless QasmSimulator', 'Noisy QasmSimulator'])
plt.title('Run time vs. shot number')