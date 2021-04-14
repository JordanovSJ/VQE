import pandas as pd

df_adapt_f_input = pd.read_csv('../../../results/iter_vqe_results/H4_adapt_vqe_eff_f_exc_r=1_09-Mar-2021.csv')
df_adapt_q_input = pd.read_csv('../../../results/iter_vqe_results/H4_adapt_vqe_q_exc_r=1_08-Mar-2021.csv')

df = {}
df['n'] = df_adapt_f_input['n']
df['q element'] = df_adapt_q_input['element']
df['f element'] = df_adapt_f_input['element']

df['q cnot'] = df_adapt_q_input['cnot_count']
df['f cnot'] = df_adapt_f_input['cnot_count']

results_df = pd.DataFrame(df)
results_df.to_csv('scripts/zhenghao/csv_folder/q_f_ansatzelements')

for idx in range(1, 20):
    n = results_df['n'][idx - 1]
    q_element = results_df['q element'][idx - 1]
    if 'd_q_exc_' in q_element:
        q_element = q_element.replace('d_q_exc_', 'd(').replace('_', ',') + ')'
    else:
        q_element = q_element.replace('s_q_exc_', 's(').replace('_', ',') + ')'
    f_element = results_df['f element'][idx - 1]
    if 'eff_d_f_exc_' in f_element:
        f_element = f_element.replace('eff_d_f_exc_', 'd(').replace('_', ',') + ')'
    else:
        f_element = f_element.replace('eff_s_f_exc_', 's(').replace('_', ',') + ')'
    q_cnot = results_df['q cnot'][idx - 1]
    f_cnot = results_df['f cnot'][idx - 1]
    cnot_diff = f_cnot - q_cnot
    print('{} & {} & {} & {} & {} & {}\\'.format(n, q_element, f_element, q_cnot, f_cnot, cnot_diff))
