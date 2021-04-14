import pandas as pd
import matplotlib.pyplot as plt
from src.backends import QiskitSimBackend
from src.iter_vqe_utils import DataUtils
from src.molecules.molecules import H4



# q exc Nelder Mead no noise 1 ansatz element, init parm = 0
df_0 = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_Nelder-Mead_no_noise_shots=1000000.0_17-Mar-2021 (22:49:16.149408).csv')

# q exc Nelder Mead no noise 1 ansatz element, init parm = -0.1
df_minus = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_Nelder-Mead_no_noise_shots=1000000.0_18-Mar-2021 (12:17:23.379625).csv')

# q exc Nelder Mead no noise 1 ansatz element, init parm = 0.1
df_plus = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_Nelder-Mead_no_noise_shots=1000000.0_18-Mar-2021 (14:21:39.661363).csv')

# q exc qiskitsim Nelder Mead 1 ansatz element, init parm = 0
df_q_qiskit_one = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_qiskitsim_Nelder-Mead_shots=1000000.0_17-Mar-2021 (22:43:47.708394).csv')


plt.figure(1)
plt.plot(df_q_qiskit_one['iteration'], df_q_qiskit_one['energy'], label='QiskitSimBackend, init par=0')
plt.plot(df_0['iteration'], df_0['energy'], label='QasmBackend, init par=0')
plt.plot(df_plus['iteration'], df_plus['energy'], label='QasmBackend, init par=0.1')
plt.plot(df_minus['iteration'], df_minus['energy'], label='QasmBackend, init par=-0.1')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Energy (Hartree)')
plt.title('VQE for one element with different init param')

exact_energy = list(df_q_qiskit_one['energy'])[-1]
chem_acc = 1e-3

plt.figure(2)
plt.plot(df_0['iteration'], abs(df_0['energy']-exact_energy), label='QasmBackend, init par=0')
plt.plot(df_plus['iteration'], abs(df_plus['energy']-exact_energy), label='QasmBackend, init par=0.1')
plt.plot(df_minus['iteration'], abs(df_minus['energy']-exact_energy), label='QasmBackend, init par=-0.1')
plt.hlines(chem_acc, 0, max(df_0['iteration']), label='chem accuracy')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Abs energy difference (Hartree)')
plt.yscale('log')
plt.title('VQE for one element with different init param')