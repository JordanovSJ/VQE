import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('../../../results/zhenghao_testing/H4_run_time_vs_ansatz_elements_shots=1000000.0.csv')
df_adapt_input = pd.read_csv('../../../results/iter_vqe_results/H4_adapt_vqe_q_exc_r=1_08-Mar-2021.csv')

cnot_count_list = [df_adapt_input['cnot_count'][i-1] for i in df['element_num']]

plt.figure(1)
plt.plot(df['element_num'], df['noisy_time'], 'x', label='Noisy')
plt.plot(df['element_num'], df['noiseless_time'], 'x', label='Noiseless')
plt.legend()
plt.xlabel('Number of Ansatz elements')
plt.ylabel('Evaluation time (s)')
plt.title('Run time of ham_expectation_value for q_exc based ansatz')

plt.figure(2)
plt.plot(cnot_count_list, df['noisy_time'], 'x', label='Noisy')
plt.plot(cnot_count_list, df['noiseless_time'], 'x', label='Noiseless')
plt.legend()
plt.xlabel('CNOT count')
plt.ylabel('Evaluation time (s)')
plt.title('Run time of ham_expectation_value for q_exc based ansatz')


