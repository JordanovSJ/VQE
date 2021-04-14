import pandas as pd
import matplotlib.pyplot as plt

df_name_list = []
df_name_list.append(
    '../../../results/zhenghao_testing/H4_vqe_eff_f_exc_p=0.0001_tcx=0_shots=1000000.0_11-Mar-2021 (00:28:59.830004).csv')
df_name_list.append(
    '../../../results/zhenghao_testing/H4_vqe_eff_f_exc_p=1e-05_tcx=0_shots=1000000.0_11-Mar-2021 (00:29:25.512922).csv')
df_name_list.append(
    '../../../results/zhenghao_testing/H4_vqe_eff_f_exc_p=1e-06_tcx=0_shots=1000000.0_11-Mar-2021 (00:30:05.450994).csv')
df_name_list.append(
    '../../../results/zhenghao_testing/H4_vqe_eff_f_exc_p=1e-06_tcx=10_shots=1000000.0_11-Mar-2021 (00:30:33.742147).csv')
df_name_list.append(
    '../../../results/zhenghao_testing/H4_vqe_q_exc_p=0.0001_tcx=0_shots=1000000.0_11-Mar-2021 (00:20:12.269867).csv')
df_name_list.append(
    '../../../results/zhenghao_testing/H4_vqe_q_exc_p=1e-05_tcx=0_shots=1000000.0_11-Mar-2021 (00:20:39.265207).csv')
df_name_list.append(
    '../../../results/zhenghao_testing/H4_vqe_q_exc_p=1e-06_tcx=0_shots=1000000.0_11-Mar-2021 (00:25:58.462368).csv')
df_name_list.append(
    '../../../results/zhenghao_testing/H4_vqe_q_exc_p=1e-06_tcx=10_shots=1000000.0_11-Mar-2021 (00:27:04.159708).csv')

df_qis = pd.read_csv(
    '../../../results/zhenghao_testing/vqe_bfgs/H4_vqe_q_exc_qiskitsim_gtol=0.001_p=1e-05_tcx=0_shots=1000000.0_11-Mar-2021 (18:03:56.357119).csv')

df_dict = {}
for df_name in df_name_list:
    df_dict[df_name] = pd.read_csv(df_name)

ground_state_E = -2.166387448527481

plt.figure(1)
plt.plot(df_qis['iteration'], df_qis['energy'], linestyle='-', label='QiskitSimBackend')
for df_name in df_dict.keys():
    df = df_dict[df_name]

    vqe_idx = df_name.index('vqe')
    exc_idx = df_name.index('exc')
    p_idx = df_name.index('p=')
    tcx_idx = df_name.index('tcx')
    shot_idx = df_name.index('shots')

    ansatz_type = df_name[vqe_idx + 4: exc_idx - 1]
    prob_2 = df_name[p_idx + 2: tcx_idx - 1]
    tcx = df_name[tcx_idx + 4: shot_idx - 1]

    if ansatz_type == 'eff_f':
        line_style = '--'
    else:
        line_style = '-'

    plt.plot(df['iteration'], df['energy'], linestyle=line_style,
             label='{},p2={},tcx={}'.format(ansatz_type, prob_2, tcx))
plt.hlines(ground_state_E, 0, max(df_qis['iteration']), label='Ground state')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Energy (Hartree)')
plt.title('VQE optimization for q_exc and f_exc under noise')
plt.show()

plt.figure(2)
plt.plot(df_qis['iteration'], abs(df_qis['energy'] - ground_state_E), linestyle='-', label='QiskitSimBackend')
for df_name in df_dict.keys():
    df = df_dict[df_name]

    vqe_idx = df_name.index('vqe')
    exc_idx = df_name.index('exc')
    p_idx = df_name.index('p=')
    tcx_idx = df_name.index('tcx')
    shot_idx = df_name.index('shots')

    ansatz_type = df_name[vqe_idx + 4: exc_idx - 1]
    prob_2 = df_name[p_idx + 2: tcx_idx - 1]
    tcx = df_name[tcx_idx + 4: shot_idx - 1]

    if ansatz_type == 'eff_f':
        line_style = '--'
    else:
        line_style = '-'

    plt.plot(df['iteration'], abs(df['energy'] - ground_state_E), linestyle=line_style,
             label='{},p2={},tcx={}'.format(ansatz_type, prob_2, tcx))

plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Difference from ground state (Hartree)')
plt.yscale('log')
plt.title('VQE convergence for q and f exc')
plt.show()
