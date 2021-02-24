import pandas as pd
import matplotlib.pyplot as plt

plt.close('all')

noisy_df = pd.read_csv('../../../results/zhenghao_testing/H4_r=1.546_noise_comparison_method=statevector_23-Feb-2021 (13:17:34.006525)')

noiseless_E = noisy_df['noiseless E']
noisy_E = noisy_df['noisy E']
time_list = noisy_df['time']

iterations= list(range(1, len(noisy_E)+1))

plt.figure(1)
plt.plot(iterations, noiseless_E)
plt.plot(iterations, noisy_E)
plt.legend(['Noiseless', 'ibmq_16_melbourne noise model'])
plt.xlabel('Number of Ansatz Elements')
plt.ylabel('Energy (Hartree)')
plt.title('Noisy comparison for ham_expectation_value for H4(1.546)')

plt.figure(2)
plt.plot(iterations, time_list)
plt.xlabel('Number of Ansatz Elements')
plt.ylabel('Time (s)')
plt.title('Run time for ham_expectation_value under noise for H4(1.546)')