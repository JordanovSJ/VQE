import pandas as pd
import numpy as np

df = pd.read_csv('../../../results/zhenghao_testing/std_dev/H4_p2=1e-06_shots=1000000.0_12-Apr-2021 (11:12:39.702543).csv')

q_energy = list(df['q_energy'])
f_energy = list(df['f_energy'])

q_std = np.std(q_energy)
f_std = np.std(f_energy)

print('q_std={}'.format(q_std))
print('f_std={}'.format(f_std))