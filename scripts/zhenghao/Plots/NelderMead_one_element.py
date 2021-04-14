import pandas as pd
import matplotlib.pyplot as plt
from src.backends import QiskitSimBackend
from src.iter_vqe_utils import DataUtils
from src.molecules.molecules import H4

q_system = H4(r=1)
num_ansatz_element = 1
df_input = pd.read_csv('../../../results/iter_vqe_results/H4_adapt_vqe_q_exc_r=1_08-Mar-2021.csv')
ansatz_state = DataUtils.ansatz_from_data_frame(df_input, q_system)
ansatz = ansatz_state.ansatz_elements[0:num_ansatz_element]


# q exc Nelder Mead no noise 1 ansatz element, init parm = 0
# df_q_qasm_one = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_Nelder-Mead_no_noise_shots=1000000.0_17-Mar-2021 (22:49:16.149408).csv')

# q exc Nelder Mead no noise 1 ansatz element, init parm = -0.1
df_q_qasm_one = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_Nelder-Mead_no_noise_shots=1000000.0_18-Mar-2021 (12:17:23.379625).csv')

# q exc Nelder Mead no noise 1 ansatz element, init parm = 0.1
# df_q_qasm_one = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_Nelder-Mead_no_noise_shots=1000000.0_18-Mar-2021 (14:21:39.661363).csv')

energy_qasm = df_q_qasm_one['energy'][0:20]
params_str_ls_qasm = df_q_qasm_one['params'][0:20]
params_list_qasm = []  # list of params [param, param ...]
params_flt_ls_ls = []  # [[param], [param]...]
for element in params_str_ls_qasm:
    element_1 = element.replace('[', '').replace(']', '')
    param = float(element_1)
    params_list_qasm.append(param)
    params_flt_ls_ls.append([param])
assert len(params_flt_ls_ls) == len(energy_qasm)

# q exc qiskitsim Nelder Mead 1 ansatz element, init parm = 0
df_q_qiskit_one = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_qiskitsim_Nelder-Mead_shots=1000000.0_17-Mar-2021 (22:43:47.708394).csv')

energy_qiskit = df_q_qiskit_one['energy']
params_str_ls_qiskit = df_q_qiskit_one['params']
params_ls_qiskit = [] # list of params [param, param ...]
for element in params_str_ls_qiskit:
    element_1 = element.replace('[', '').replace(']', '')
    param = float(element_1)
    params_ls_qiskit.append(param)
assert len(params_ls_qiskit) == len(energy_qiskit)

energy_qiskit_compare = []
for var_pars in params_flt_ls_ls:
    exp_value = QiskitSimBackend.ham_expectation_value(var_parameters=var_pars, ansatz=ansatz, q_system=q_system)
    energy_qiskit_compare.append(exp_value)


plt.figure(1)
plt.plot(params_list_qasm, energy_qasm, 'bx', label='QasmBackend')
plt.plot(params_list_qasm, energy_qiskit_compare, 'ro', label='QiskitBackend')
plt.xlabel('theta 1')
plt.ylabel('Energy')
plt.ylim([-2.125, -2.095])
plt.legend()
plt.title('VQE for one element on QasmBackend')

plt.figure(2)
plt.plot(params_ls_qiskit, energy_qiskit, 'x')
plt.xlabel('theta 1')
plt.ylabel('Energy')
plt.ylim([-2.125, -2.095])
plt.legend()
plt.title('VQE for one element on QiskitSimBackend')
