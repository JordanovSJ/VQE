import pandas as pd
import matplotlib.pyplot as plt

plt.close('all')

noisy_df = pd.read_csv('../../../results/zhenghao_testing/LiH_r=3.25_noise_comparison_21-Feb-2021')
noiseless_df = pd.read_csv('../../../results/iter_vqe_results/LiH_iqeb_q_exc_r=3.25_19-Nov-2020.csv')

myfile = open('../../../results/logs/17-Feb-2021 (21:33:51.412616).txt')
log_contents = myfile.read()
myfile.close()
log_lines = log_contents.splitlines()
sub = 'time = '
time_list = []

for line in log_lines:
    if sub in line:
        index1 = line.index(sub)+7
        index2 = line.index('s,')
        time_list.append(float(line[index1:index2]))

noisy_E = noisy_df['noisy_E']
noiseless_E = noiseless_df['E']
iterations = list(range(1, len(noisy_E)+1))

plt.figure(1)
plt.plot(iterations, noiseless_E)
plt.plot(iterations, noisy_E)
plt.legend(['Noiseless', 'ibmq_16_melbourne noise model'])
plt.xlabel('Number of Ansatz Elements')
plt.ylabel('Energy (Hartree)')
plt.title('Noisy comparison for ham_expectation_value for LiH(r=3.25)')

plt.figure(2)
plt.plot(iterations, time_list)
plt.xlabel('Number of Ansatz Elements')
plt.ylabel('Time (s)')
plt.title('Run time for ham_expectation_value under noise for LiH(r=3.25)')