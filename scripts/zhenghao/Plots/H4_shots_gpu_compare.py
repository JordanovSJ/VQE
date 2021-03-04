import pandas as pd
import matplotlib.pyplot as plt

plt.close('all')

cpu_df = pd.read_csv('../../../results/zhenghao_testing/'
                     'H4_r=1.546_iqeb_ansatz_compare_method=statevector_04-Mar-2021 '
                     '(12:13:52.189604)')

gpu_df = pd.read_csv('../../../results/zhenghao_testing/'
                     'H4_r=1.546_iqeb_ansatz_compare_method=statevector_gpu_04-Mar-2021 '
                     '(12:13:52.189604)')

shots = cpu_df['n_shots']
ref_E = cpu_df['ref E']

noiseless_cpu = cpu_df['noiseless E']
noiseless_cpu_time = cpu_df['noiseless time']
noisy_cpu = cpu_df['noisy E']
noisy_cpu_time = cpu_df['noisy time']

noiseless_gpu = gpu_df['noiseless E']
noiseless_gpu_time = gpu_df['noiseless time']
noisy_gpu = gpu_df['noisy E']
noisy_gpu_time = gpu_df['noisy time']

plt.figure(1)
plt.plot(shots, ref_E)
plt.plot(shots, noiseless_cpu, 'x')
plt.plot(shots, noiseless_gpu, 'x')
plt.xlabel('Number of shots')
plt.ylabel('Hamiltonian expectation value')
plt.xscale('log')
plt.legend(['Exact', 'noiseless statevector', 'noiseless statevector_gpu'])
plt.title('Hamiltonian expectation value against shot number')

plt.figure(2)
plt.plot(shots, abs(noiseless_cpu-ref_E), 'x')
plt.plot(shots, abs(noiseless_gpu-ref_E), 'x')
plt.xlabel('Number of shots')
plt.ylabel('|delta E|')
plt.yscale('log')
plt.xscale('log')
plt.legend(['noiseless statevector', 'noiseless statevector_gpu'])
plt.title('ham expectation value difference from exact against shot number')

plt.figure(3)
plt.plot(shots, noiseless_cpu_time)
plt.plot(shots, noiseless_gpu_time)
plt.plot(shots, noisy_cpu_time)
plt.plot(shots, noisy_gpu_time)
plt.legend(['noiseless statevector', 'noiseless statevector_gpu',
            'ibmq melbourne statevector', 'ibmq melbourne statevector_gpu'])
plt.title('Run time against shot number')
plt.xlabel('Number of shots')
plt.xscale('log')
plt.ylabel('Run time (s)')



