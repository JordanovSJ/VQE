import pandas as pd
import matplotlib.pyplot as plt
from src.molecules.molecules import H4

molecule = H4(r=1)
ground_state = molecule.fci_energy
chem_acc = 1e-3

def select_value(df_series):

    to_choose = min(df_series)

    # to_choose = list(df_series)[-1]

    return to_choose


df_adapt_f_input = pd.read_csv('../../../results/iter_vqe_results/H4_adapt_vqe_eff_f_exc_r=1_09-Mar-2021.csv')
df_adapt_q_input = pd.read_csv('../../../results/iter_vqe_results/H4_adapt_vqe_q_exc_r=1_08-Mar-2021.csv')

# 1 element
# qexc, p2=1e-4, init param = 0.0001
df_q_4_oneelement = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=0_1_elements_shots=1000000.0_21-Mar-2021 (19:21:37.435522).csv')

# qexc, p2=1e-6, init param = 0.0001
df_q_6_oneelement = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=1_tcx=1e-06_0_elements_shots=1000000.0_19-Mar-2021 (13:03:37.248538).csv')

# fexc, p2=1e-6, init param = 0.0001
df_f_6_oneelement = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=1e-06_tcx=0_1_elements_shots=1000000.0_21-Mar-2021 (12:07:46.393465).csv')

# fexc, p2=1e-4, init param = 0.0001
df_f_4_oneelement = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=0_1_elements_shots=1000000.0_21-Mar-2021 (17:39:40.980424).csv')

# 5 elements
# qexc, p2=1e-6, init param = 0.0001
df_q_6_fiveelement = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=5_tcx=1e-06_0_elements_shots=1000000.0_19-Mar-2021 (13:26:58.438456).csv')

# qexc, p2=1e-4, init param = 0.0001
df_q_4_fiveelement = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=0_5_elements_shots=1000000.0_19-Mar-2021 (16:00:50.233544).csv')

# fexc, p2=1e-6, init param = 0.0001
df_f_6_fiveelement = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=1e-06_tcx=0_5_elements_shots=1000000.0_21-Mar-2021 (12:09:47.758194).csv')

# fexc, p2=1e-4, init param = 0.0001
df_f_4_fiveelement = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=0_5_elements_shots=1000000.0_21-Mar-2021 (17:40:16.619704).csv')

# qexc, p2=1e-4, tcx=10, init par = 1e-4
df_q_4_tcx_fiveelement = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=10_5_elements_shots=1000000.0_22-Mar-2021 (23:25:49.320787).csv')

# fexc, p2=1e-4, tcx=10, init par =1e-4
df_f_4_tcx_fiveelement = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=10_5_elements_shots=1000000.0_23-Mar-2021 (11:34:46.705074).csv')

# 6 elements
filename_6elem = {
    'q_exc, p2=1e-4': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=0_6_elements_shots=1000000.0_09-Apr-2021 (04:27:37.882731).csv',
    'f_exc, p2=1e-4': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=0_6_elements_shots=1000000.0_07-Apr-2021 (11:58:52.250229).csv',
    'q_exc, p2=1e-6': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=1e-06_tcx=0_6_elements_shots=1000000.0_09-Apr-2021 (06:08:29.375977).csv',
    'f_exc, p2=1e-6': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=1e-06_tcx=0_6_elements_shots=1000000.0_07-Apr-2021 (11:59:28.977757).csv',
}
df_6elem = {}
for df_label in filename_6elem:
    df_6elem[df_label] = pd.read_csv('../../../' + filename_6elem[df_label])


# 7 elements
filename_7elem = {
    'q_exc, p2=1e-4': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=0_7_elements_shots=1000000.0_04-Apr-2021 (10:44:26.092607).csv',
    'f_exc, p2=1e-4': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=0_7_elements_shots=1000000.0_03-Apr-2021 (10:34:53.534544).csv',
    'q_exc, p2=1e-6': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=1e-06_tcx=0_7_elements_shots=1000000.0_03-Apr-2021 (08:40:34.254973).csv',
    'f_exc, p2=1e-6': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=1e-06_tcx=0_7_elements_shots=1000000.0_02-Apr-2021 (12:43:28.822851).csv',
}
df_7elem = {}
for df_label in filename_7elem:
    df_7elem[df_label] = pd.read_csv('../../../' + filename_7elem[df_label])

# 9 elements
filename_dict_9_elem = {
    'q_exc, p2=1e-4': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=0_9_elements_shots=1000000.0_04-Apr-2021 (18:13:43.888979).csv',
    'f_exc, p2=1e-4': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=0_9_elements_shots=1000000.0_03-Apr-2021 (21:08:52.889638).csv',
    'q_exc, p2=1e-6': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=1e-06_tcx=0_9_elements_shots=1000000.0_03-Apr-2021 (13:55:32.808408).csv',
    'f_exc, p2=1e-6': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=1e-06_tcx=0_9_elements_shots=1000000.0_02-Apr-2021 (21:15:07.754071).csv',
}
df_9elem = {}
for df_label in filename_dict_9_elem:
    df_9elem[df_label] = pd.read_csv('../../../' + filename_dict_9_elem[df_label])


# 11 elements
# qexc, p2=1e-6, init par = -0.1
df_q_6_11element = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=1e-06_tcx=0_shots=1000000.0_18-Mar-2021 (19:12:16.969257).csv')
# fexc, p2=1e-6, init par = -0.1
df_f_6_11element = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=1e-06_tcx=0_shots=1000000.0_18-Mar-2021 (19:13:52.870310).csv')
# qexc, p2=1e-4, init par = 0.0001
df_q_4_11element = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=0_11_elements_shots=1000000.0_21-Mar-2021 (22:53:09.328452).csv')
# fexc, p2=1e-4, init par = 0.0001
df_f_4_11element = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=0_11_elements_shots=1000000.0_22-Mar-2021 (22:47:33.008108).csv')
# qexc, p2=1e-4, tcx=10, init par = 0.0001
df_q_4_tcx_11element = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=10_11_elements_shots=1000000.0_24-Mar-2021 (10:37:17.543387).csv')
# fexc, p2=1e-4, tcx=10, init par = 0.0001
df_f_4_tcx_11element = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=10_11_elements_shots=1000000.0_24-Mar-2021 (10:38:55.731061).csv')

# 13 elements
filename_dict_13elem = {
    'q_exc, p2=1e-4': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=0_13_elements_shots=1000000.0_03-Apr-2021 (10:37:03.492609).csv',
    'f_exc, p2=1e-4': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=0_13_elements_shots=1000000.0_05-Apr-2021 (10:05:12.465140).csv',
    'q_exc, p2=1e-6': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=1e-06_tcx=0_13_elements_shots=1000000.0_05-Apr-2021 (10:10:44.820947).csv',
    'f_exc, p2=1e-6': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=1e-06_tcx=0_13_elements_shots=1000000.0_05-Apr-2021 (10:14:44.604619).csv',
}

df_13elem = {}
for df_label in filename_dict_13elem:
    df_13elem[df_label] = pd.read_csv('../../../' + filename_dict_13elem[df_label])

# 15 elements
# qexc, p2=1e-6, 11 inspired pars and rest 0.01
df_q_6_15element = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=1e-06_tcx=0_15_elements_shots=1000000.0_22-Mar-2021 (00:12:59.384017).csv')

# fexc, p2=1e-6, 11 inspired pars and rest 0.01
df_f_6_15element = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=1e-06_tcx=0_15_elements_shots=1000000.0_22-Mar-2021 (09:32:43.987574).csv')

# qexc, p2=1e-4, 11 inspired pars and rest 0.01
df_q_4_15element = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=0_15_elements_shots=1000000.0_22-Mar-2021 (23:00:52.350533).csv')

# fexc, p2=1e-4, 11 inspired pars and rest 0.01
df_f_4_15element = pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=0_15_elements_shots=1000000.0_23-Mar-2021 (21:13:58.220540).csv')


df_8elem = {
    'q_exc, p2=1e-4': pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=0_8_elements_shots=1000000.0_09-Apr-2021 (11:51:21.348700).csv'),
    'f_exc, p2=1e-4': pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=0_8_elements_shots=1000000.0_07-Apr-2021 (19:36:01.431768).csv'),
    'q_exc, p2=1e-6': pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=1e-06_tcx=0_8_elements_shots=1000000.0_09-Apr-2021 (13:35:59.559655).csv'),
    'f_exc, p2=1e-6': pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=1e-06_tcx=0_8_elements_shots=1000000.0_07-Apr-2021 (20:01:20.841731).csv')
}

df_10elem = {
    'q_exc, p2=1e-4': pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=0_10_elements_shots=1000000.0_10-Apr-2021 (00:22:48.535296).csv'),
    'f_exc, p2=1e-4': pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=0_10_elements_shots=1000000.0_08-Apr-2021 (08:06:54.688568).csv'),
    'q_exc, p2=1e-6': pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=1e-06_tcx=0_10_elements_shots=1000000.0_10-Apr-2021 (02:12:43.640620).csv'),
    'f_exc, p2=1e-6': pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=1e-06_tcx=0_10_elements_shots=1000000.0_11-Apr-2021 (16:11:38.117366).csv')
}

df_12elem = {
    'q_exc, p2=1e-4': pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=0_12_elements_shots=1000000.0_12-Apr-2021 (19:40:17.656100).csv'),
    'f_exc, p2=1e-4': pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=0_12_elements_shots=1000000.0_12-Apr-2021 (08:14:19.764343).csv'),
    'q_exc, p2=1e-6': pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=1e-06_tcx=0_12_elements_shots=1000000.0_08-Apr-2021 (16:07:55.134903).csv'),
    'f_exc, p2=1e-6': pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=1e-06_tcx=0_12_elements_shots=1000000.0_07-Apr-2021 (22:04:35.901459).csv')
}

df_14elem = {
    'q_exc, p2=1e-4': pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=0_14_elements_shots=1000000.0_11-Apr-2021 (15:33:46.244151).csv'),
    'f_exc, p2=1e-4': pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=0_14_elements_shots=1000000.0_11-Apr-2021 (06:55:52.739082).csv'),
    'q_exc, p2=1e-6': pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=1e-06_tcx=0_14_elements_shots=1000000.0_07-Apr-2021 (09:58:43.326530).csv'),
    'f_exc, p2=1e-6': pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=1e-06_tcx=0_14_elements_shots=1000000.0_07-Apr-2021 (04:13:24.881636).csv')
}

df_16elem = {
    'q_exc, p2=1e-4': pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=0_16_elements_shots=1000000.0_10-Apr-2021 (12:20:10.804422).csv'),
    'f_exc, p2=1e-4': pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=0_16_elements_shots=1000000.0_09-Apr-2021 (07:54:14.441053).csv'),
    'q_exc, p2=1e-6': pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=1e-06_tcx=0_16_elements_shots=1000000.0_06-Apr-2021 (22:03:06.233869).csv'),
    'f_exc, p2=1e-6': pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=1e-06_tcx=0_16_elements_shots=1000000.0_06-Apr-2021 (12:28:31.699328).csv'),
    'q_exc, p2=1e-4, tcx=10': pd.read_csv('../../../results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=10_16_elements_shots=1000000.0_13-Apr-2021 (07:25:58.710294).csv'),
    'f_exc, p2=1e-4, tcx=10': pd.read_csv('../../../results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=10_16_elements_shots=1000000.0_13-Apr-2021 (00:16:48.559473).csv')
}

num_element_list = [1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
q_exc_6_list = [
    select_value(df_q_6_oneelement['energy']),
    select_value(df_q_6_fiveelement['energy']),
    select_value(df_6elem['q_exc, p2=1e-6']['energy']),
    select_value(df_7elem['q_exc, p2=1e-6']['energy']),
    select_value(df_8elem['q_exc, p2=1e-6']['energy']),
    select_value(df_9elem['q_exc, p2=1e-6']['energy']),
    select_value(df_10elem['q_exc, p2=1e-6']['energy']),
    select_value(df_q_6_11element['energy']),
    select_value(df_12elem['q_exc, p2=1e-6']['energy']),
    select_value(df_13elem['q_exc, p2=1e-6']['energy']),
    select_value(df_14elem['q_exc, p2=1e-6']['energy']),
    select_value(df_q_6_15element['energy']),
    select_value(df_16elem['q_exc, p2=1e-6']['energy']),
]
q_exc_4_list = [
    select_value(df_q_4_oneelement['energy']),
    select_value(df_q_4_fiveelement['energy']),
    select_value(df_6elem['q_exc, p2=1e-4']['energy']),
    select_value(df_7elem['q_exc, p2=1e-4']['energy']),
    select_value(df_8elem['q_exc, p2=1e-4']['energy']),
    select_value(df_9elem['q_exc, p2=1e-4']['energy']),
    select_value(df_10elem['q_exc, p2=1e-4']['energy']),
    select_value(df_q_4_11element['energy']),
    select_value(df_12elem['q_exc, p2=1e-4']['energy']),
    select_value(df_13elem['q_exc, p2=1e-4']['energy']),
    select_value(df_14elem['q_exc, p2=1e-4']['energy']),
    select_value(df_q_4_15element['energy']),
    select_value(df_16elem['q_exc, p2=1e-4']['energy']),
]
f_exc_6_list = [
    select_value(df_f_6_oneelement['energy']),
    select_value(df_f_6_fiveelement['energy']),
    select_value(df_6elem['f_exc, p2=1e-6']['energy']),
    select_value(df_7elem['f_exc, p2=1e-6']['energy']),
    select_value(df_8elem['f_exc, p2=1e-6']['energy']),
    select_value(df_9elem['f_exc, p2=1e-6']['energy']),
    select_value(df_10elem['f_exc, p2=1e-6']['energy']),
    select_value(df_f_6_11element['energy']),
    select_value(df_12elem['f_exc, p2=1e-6']['energy']),
    select_value(df_13elem['f_exc, p2=1e-6']['energy']),
    select_value(df_14elem['f_exc, p2=1e-6']['energy']),
    select_value(df_f_6_15element['energy']),
    select_value(df_16elem['f_exc, p2=1e-6']['energy']),
]
f_exc_4_list = [
    select_value(df_f_4_oneelement['energy']),
    select_value(df_f_4_fiveelement['energy']),
    select_value(df_6elem['f_exc, p2=1e-4']['energy']),
    select_value(df_7elem['f_exc, p2=1e-4']['energy']),
    select_value(df_8elem['f_exc, p2=1e-4']['energy']),
    select_value(df_9elem['f_exc, p2=1e-4']['energy']),
    select_value(df_10elem['f_exc, p2=1e-4']['energy']),
    select_value(df_f_4_11element['energy']),
    select_value(df_12elem['f_exc, p2=1e-4']['energy']),
    select_value(df_13elem['f_exc, p2=1e-4']['energy']),
    select_value(df_14elem['f_exc, p2=1e-4']['energy']),
    select_value(df_f_4_15element['energy']),
    select_value(df_16elem['f_exc, p2=1e-4']['energy']),
]

tcx_num_elem_list = [5, 11, 16]

q_exc_4_tcx_list = [
    select_value(df_q_4_tcx_fiveelement['energy']),
    select_value(df_q_4_tcx_11element['energy']),
    select_value(df_16elem['q_exc, p2=1e-4, tcx=10']['energy'])
]

f_exc_4_tcx_list = [
    select_value(df_f_4_tcx_fiveelement['energy']),
    select_value(df_f_4_tcx_11element['energy']),
    select_value(df_16elem['f_exc, p2=1e-4, tcx=10']['energy'])
]

plt.figure(1)
plt.plot(df_adapt_q_input['n'], df_adapt_q_input['E'], label='q_exc, exact')
plt.plot(df_adapt_f_input['n'], df_adapt_f_input['E'], label='f_exc, exact')
plt.plot(num_element_list[0:len(q_exc_6_list)], q_exc_6_list, 'x',
         label='q_exc, p_2=p_m=1e-6')
plt.plot(num_element_list[0:len(f_exc_6_list)], f_exc_6_list, 'x',
         label= 'f_exc, p_2=p_m=1e-6')
plt.plot(num_element_list[0:len(q_exc_4_list)], q_exc_4_list, 'x',
         label='q_exc, p_2=p_m=1e-4')
plt.plot(num_element_list[0:len(f_exc_4_list)], f_exc_4_list, 'x',
         label='f_exc, p_2=p_m=1e-4')
plt.plot(tcx_num_elem_list, q_exc_4_tcx_list, 'x',
         label='q_exc, p_2=p_m=1e-4, tcx=10')
plt.plot(tcx_num_elem_list, f_exc_4_tcx_list, 'x',
         label='f_exc, p_2=p_m=1e-4, tcx=10')

xint = range(0, 17)
plt.xticks(xint)

plt.xlabel('Number of Ansatz elements')
plt.ylabel('Energy (Hartree)')
plt.legend()
plt.title('Fermionic vs Qubit excitation based iterative VQE with H4(r=1)')

plt.figure(0)
plt.plot(df_adapt_f_input['n'], df_adapt_f_input['cnot_count'], label='F Ansatz')
plt.plot(df_adapt_q_input['n'], df_adapt_q_input['cnot_count'], label='Q Ansatz')
plt.xlabel('Number of elements')
plt.ylabel('CNOT count')
plt.legend()
xint = range(0, 20)
plt.xticks(xint)
plt.title('CNOT count vs number of elements')

diff_cnot = df_adapt_f_input['cnot_count']-df_adapt_q_input['cnot_count']
diff_cnot_noisy = [diff_cnot[i-1] for i in num_element_list]
diff_cnot_tcx = [diff_cnot[i-1] for i in tcx_num_elem_list]

plt.figure(2)
plt.plot(diff_cnot, df_adapt_q_input['E'], label='q_exc, exact')
plt.plot(diff_cnot, df_adapt_f_input['E'], label='f_exc, exact')
plt.plot(diff_cnot_noisy[0:len(q_exc_6_list)], q_exc_6_list, 'x',
         label='q_exc, p_2=p_m=1e-6')
plt.plot(diff_cnot_noisy[0:len(f_exc_6_list)], f_exc_6_list, 'x',
         label= 'f_exc, p_2=p_m=1e-6')
plt.plot(diff_cnot_noisy[0:len(q_exc_4_list)], q_exc_4_list, 'x',
         label='q_exc, p_2=p_m=1e-4')
plt.plot(diff_cnot_noisy[0:len(f_exc_4_list)], f_exc_4_list, 'x',
         label='f_exc, p_2=p_m=1e-4')
plt.plot(diff_cnot_tcx, q_exc_4_tcx_list, 'x',
         label='q_exc, p_2=p_m=1e-4, tcx=10')
plt.plot(diff_cnot_tcx, f_exc_4_tcx_list, 'x',
         label='f_exc, p_2=p_m=1e-4, tcx=10')
plt.xlabel('Difference in CNOT count')
plt.ylabel('Energy (Hartree)')
# plt.legend()
plt.title('Fermionic vs Qubit excitation based iterative VQE with H4(r=1)')

plt.figure(3)
plt.plot(df_adapt_q_input['n'][0:16], df_adapt_q_input['E'][0:16]-ground_state, label='q_exc, exact')
plt.plot(df_adapt_f_input['n'][0:16], df_adapt_f_input['E'][0:16]-ground_state, label='f_exc, exact')
plt.plot(num_element_list[0:len(q_exc_6_list)], [abs(x-ground_state) for x in q_exc_6_list], 'x',
         label='q_exc, p_2=p_m=1e-6')
plt.plot(num_element_list[0:len(f_exc_6_list)], [abs(x-ground_state) for x in f_exc_6_list], 'x',
         label= 'f_exc, p_2=p_m=1e-6')
plt.plot(num_element_list[0:len(q_exc_4_list)], [abs(x-ground_state) for x in q_exc_4_list], 'x',
         label='q_exc, p_2=p_m=1e-4')
plt.plot(num_element_list[0:len(f_exc_4_list)], [abs(x-ground_state) for x in f_exc_4_list], 'x',
         label='f_exc, p_2=p_m=1e-4')
plt.plot(tcx_num_elem_list, [x-ground_state for x in q_exc_4_tcx_list], 'x',
         label='q_exc, p_2=p_m=1e-4, tcx=10')
plt.plot(tcx_num_elem_list, [x-ground_state for x in f_exc_4_tcx_list], 'x',
         label='f_exc, p_2=p_m=1e-4, tcx=10')
plt.hlines(chem_acc, 0, 17, label='chem accuracy')
plt.yscale('log')

xint = range(0, 17)
plt.xticks(xint)

plt.xlabel('Number of Ansatz elements')
plt.ylabel('Energy difference (Hartree)')
plt.legend()
plt.title('Fermionic vs Qubit excitation based iterative VQE with H4(r=1)')

plt.figure(4)
plt.plot(diff_cnot[0:16], df_adapt_q_input['E'][0:16]-ground_state, label='q_exc, exact')
plt.plot(diff_cnot[0:16], df_adapt_f_input['E'][0:16]-ground_state, label='f_exc, exact')
plt.plot(diff_cnot_noisy[0:len(q_exc_6_list)], [abs(x-ground_state) for x in q_exc_6_list], 'x',
         label='q_exc, p_2=p_m=1e-6')
plt.plot(diff_cnot_noisy[0:len(f_exc_6_list)], [abs(x-ground_state) for x in f_exc_6_list], 'x',
         label= 'f_exc, p_2=p_m=1e-6')
plt.plot(diff_cnot_noisy[0:len(q_exc_4_list)], [abs(x-ground_state) for x in q_exc_4_list], 'x',
         label='q_exc, p_2=p_m=1e-4')
plt.plot(diff_cnot_noisy[0:len(f_exc_4_list)], [abs(x-ground_state) for x in f_exc_4_list], 'x',
         label='f_exc, p_2=p_m=1e-4')
plt.plot(diff_cnot_tcx, [x-ground_state for x in q_exc_4_tcx_list], 'x',
         label='q_exc, p_2=p_m=1e-4, tcx=10')
plt.plot(diff_cnot_tcx, [x-ground_state for x in f_exc_4_tcx_list], 'x',
         label='f_exc, p_2=p_m=1e-4, tcx=10')
plt.hlines(chem_acc, 0, max(diff_cnot_noisy), label='chem accuracy')
plt.yscale('log')
plt.xlabel('Difference in CNOT count')
plt.ylabel('Energy difference (Hartree)')
# plt.legend()
plt.title('Fermionic vs Qubit excitation based iterative VQE with H4(r=1)')


q_exact_list = []
f_exact_list = []

for num_elem in num_element_list:
    q_exact_list.append(df_adapt_q_input['E'][num_elem-1])
    f_exact_list.append(df_adapt_f_input['E'][num_elem-1])

plt.figure(5)
plt.plot(num_element_list[0:len(q_exc_6_list)], [abs(x-y) for x,y in zip(q_exc_6_list, q_exact_list)], 'x',
         label='q_exc, p_2=p_m=1e-6')
plt.plot(num_element_list[0:len(f_exc_6_list)], [abs(x-y) for x,y in zip(f_exc_6_list, f_exact_list)], 'x',
         label= 'f_exc, p_2=p_m=1e-6')
plt.plot(num_element_list[0:len(q_exc_4_list)], [abs(x-y) for x,y in zip(q_exc_4_list, q_exact_list)], 'x',
         label='q_exc, p_2=p_m=1e-4')
plt.plot(num_element_list[0:len(f_exc_4_list)], [abs(x-y) for x,y in zip(f_exc_4_list, f_exact_list)], 'x',
         label='f_exc, p_2=p_m=1e-4')
plt.hlines(chem_acc, 0, 17, label='chem accuracy')
plt.yscale('log')

xint = range(0, 17)
plt.xticks(xint)

plt.xlabel('Number of Ansatz elements')
plt.ylabel('Energy difference from qiskitsim (Hartree)')
plt.legend()
plt.title('Fermionic vs Qubit excitation based iterative VQE with H4(r=1)')



