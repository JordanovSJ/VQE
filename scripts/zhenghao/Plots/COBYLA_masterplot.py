import pandas as pd
import matplotlib.pyplot as plt
from src.molecules.molecules import H4

h4 = H4(r=1)
ground_state = h4.fci_energy
chem_acc = 1e-3


filename_6elem = {
    'q_exc, p2=1e-4': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=0_6_elements_shots=1000000.0_09-Apr-2021 (04:27:37.882731).csv',
    'f_exc, p2=1e-4': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=0_6_elements_shots=1000000.0_07-Apr-2021 (11:58:52.250229).csv',
    'q_exc, p2=1e-6': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=1e-06_tcx=0_6_elements_shots=1000000.0_09-Apr-2021 (06:08:29.375977).csv',
    'f_exc, p2=1e-6': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=1e-06_tcx=0_6_elements_shots=1000000.0_07-Apr-2021 (11:59:28.977757).csv',
}

filename_dict_7elem = {
    'q_exc, p2=1e-4': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=0_7_elements_shots=1000000.0_04-Apr-2021 (10:44:26.092607).csv',
    'f_exc, p2=1e-4': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=0_7_elements_shots=1000000.0_03-Apr-2021 (10:34:53.534544).csv',
    'q_exc, p2=1e-6': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=1e-06_tcx=0_7_elements_shots=1000000.0_03-Apr-2021 (08:40:34.254973).csv',
    'f_exc, p2=1e-6': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=1e-06_tcx=0_7_elements_shots=1000000.0_02-Apr-2021 (12:43:28.822851).csv',
}

filename_dict_8elem = {
    'q_exc, p2=1e-4': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=0_8_elements_shots=1000000.0_09-Apr-2021 (11:51:21.348700).csv',
    'f_exc, p2=1e-4': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=0_8_elements_shots=1000000.0_07-Apr-2021 (19:36:01.431768).csv',
    'q_exc, p2=1e-6': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=1e-06_tcx=0_8_elements_shots=1000000.0_09-Apr-2021 (13:35:59.559655).csv',
    'f_exc, p2=1e-6': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=1e-06_tcx=0_8_elements_shots=1000000.0_07-Apr-2021 (20:01:20.841731).csv',
    # 'q_exc, p2=1e-4, tcx=10': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=10_8_elements_shots=1000000.0_13-Apr-2021 (21:51:11.426095).csv',
    # 'f_exc, p2=1e-4, tcx=10': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=10_8_elements_shots=1000000.0_14-Apr-2021 (09:00:27.605283).csv',
}

filename_dict_9_elem = {
    'q_exc, p2=1e-4': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=0_9_elements_shots=1000000.0_04-Apr-2021 (18:13:43.888979).csv',
    'f_exc, p2=1e-4': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=0_9_elements_shots=1000000.0_03-Apr-2021 (21:08:52.889638).csv',
    'q_exc, p2=1e-6': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=1e-06_tcx=0_9_elements_shots=1000000.0_03-Apr-2021 (13:55:32.808408).csv',
    'f_exc, p2=1e-6': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=1e-06_tcx=0_9_elements_shots=1000000.0_02-Apr-2021 (21:15:07.754071).csv',
}

filename_dict_10elem = {
    'q_exc, p2=1e-4': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=0_10_elements_shots=1000000.0_10-Apr-2021 (00:22:48.535296).csv',
    'f_exc, p2=1e-4': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=0_10_elements_shots=1000000.0_08-Apr-2021 (08:06:54.688568).csv',
    'q_exc, p2=1e-6': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=1e-06_tcx=0_10_elements_shots=1000000.0_10-Apr-2021 (02:12:43.640620).csv',
    'f_exc, p2=1e-6': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=1e-06_tcx=0_10_elements_shots=1000000.0_11-Apr-2021 (16:11:38.117366).csv',
    'q_exc, p2=1e-8': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=1e-08_tcx=0_10_elements_shots=1000000.0_18-Apr-2021 (10:56:35.338360).csv',
    'f_exc, p2=1e-8': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=1e-08_tcx=0_10_elements_shots=1000000.0_18-Apr-2021 (10:58:32.582744).csv',
    'q_exc, shot noise': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0_tcx=0_10_elements_shots=1000000.0_23-Apr-2021 (11:10:47.342777).csv',
    'f_exc, shot noise': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0_tcx=0_10_elements_shots=1000000.0_24-Apr-2021 (10:46:07.236394).csv'
}

filename_dict_11elem = {
    'q_exc, p2=1e-4': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=0_11_elements_shots=1000000.0_21-Mar-2021 (22:53:09.328452).csv',
    'f_exc, p2=1e-4': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=0_11_elements_shots=1000000.0_22-Mar-2021 (22:47:33.008108).csv',
    'q_exc, p2=1e-6': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=1e-06_tcx=0_11_elements_shots=1000000.0_18-Mar-2021 (19:12:16.969257).csv',
    'f_exc, p2=1e-6': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=1e-06_tcx=0_11_elements_shots=1000000.0_18-Mar-2021 (19:13:52.870310).csv',
    'q_exc, p2=1e-8': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=1e-08_tcx=0_11_elements_shots=1000000.0_18-Apr-2021 (19:44:30.351709).csv',
    'f_exc, p2=1e-8': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=1e-08_tcx=0_11_elements_shots=1000000.0_18-Apr-2021 (20:41:59.872653).csv',
    'q_exc, shot noise': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0_tcx=0_11_elements_shots=1000000.0_23-Apr-2021 (16:09:21.945255).csv',
    'f_exc, shot noise': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0_tcx=0_11_elements_shots=1000000.0_24-Apr-2021 (16:55:26.946057).csv',
    # 'q_exc, p2=1e-4, tcx=10': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=10_11_elements_shots=1000000.0_24-Mar-2021 (10:37:17.543387).csv',
    # 'f_exc, p2=1e-4, tcx=10': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=10_11_elements_shots=1000000.0_24-Mar-2021 (10:38:55.731061).csv',
}

filename_dict_12elem = {
    'q_exc, p2=1e-4': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=0_12_elements_shots=1000000.0_12-Apr-2021 (19:40:17.656100).csv',
    'f_exc, p2=1e-4': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=0_12_elements_shots=1000000.0_12-Apr-2021 (08:14:19.764343).csv',
    'q_exc, p2=1e-6': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=1e-06_tcx=0_12_elements_shots=1000000.0_08-Apr-2021 (16:07:55.134903).csv',
    'f_exc, p2=1e-6': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=1e-06_tcx=0_12_elements_shots=1000000.0_07-Apr-2021 (22:04:35.901459).csv',
    'q_exc, p2=1e-8': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=1e-08_tcx=0_12_elements_shots=1000000.0_19-Apr-2021 (05:13:15.040395).csv',
    'f_exc, p2=1e-8': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=1e-08_tcx=0_12_elements_shots=1000000.0_19-Apr-2021 (06:54:41.162343).csv',
    'q_exc, shot noise': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0_tcx=0_12_elements_shots=1000000.0_23-Apr-2021 (21:36:12.263282).csv',
    'f_exc, shot noise': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0_tcx=0_12_elements_shots=1000000.0_24-Apr-2021 (23:18:42.214580).csv',
}


# 13 elements
filename_dict_13elem = {
    'q_exc, p2=1e-4': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=0_13_elements_shots=1000000.0_03-Apr-2021 (10:37:03.492609).csv',
    'f_exc, p2=1e-4': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=0_13_elements_shots=1000000.0_05-Apr-2021 (10:05:12.465140).csv',
    'q_exc, p2=1e-6': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=1e-06_tcx=0_13_elements_shots=1000000.0_05-Apr-2021 (10:10:44.820947).csv',
    'f_exc, p2=1e-6': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=1e-06_tcx=0_13_elements_shots=1000000.0_05-Apr-2021 (10:14:44.604619).csv',
    'q_exc, p2=1e-8': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=1e-08_tcx=0_13_elements_shots=1000000.0_19-Apr-2021 (15:48:47.346622).csv',
    'f_exc, p2=1e-8': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=1e-08_tcx=0_13_elements_shots=1000000.0_19-Apr-2021 (18:54:37.941070).csv',
    'q_exc, shot noise': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0_tcx=0_13_elements_shots=1000000.0_24-Apr-2021 (04:03:57.720705).csv',
    'f_exc, shot noise': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0_tcx=0_13_elements_shots=1000000.0_25-Apr-2021 (07:25:40.507389).csv',
}

filename_dict_14elem = {
    'q_exc, p2=1e-4': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=0_14_elements_shots=1000000.0_11-Apr-2021 (15:33:46.244151).csv',
    'f_exc, p2=1e-4': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=0_14_elements_shots=1000000.0_11-Apr-2021 (06:55:52.739082).csv',
    # 'q_exc, p2=1e-4, tcx=10': '',
    # 'f_exc, p2=1e-4, tcx=10': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=10_14_elements_shots=1000000.0_13-Apr-2021 (18:04:46.906607).csv',
    'q_exc, p2=1e-6': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=1e-06_tcx=0_14_elements_shots=1000000.0_07-Apr-2021 (09:58:43.326530).csv',
    'f_exc, p2=1e-6': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=1e-06_tcx=0_14_elements_shots=1000000.0_07-Apr-2021 (04:13:24.881636).csv'
}

filename_dict_16elem = {
    'q_exc, p2=1e-4': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=0_16_elements_shots=1000000.0_10-Apr-2021 (12:20:10.804422).csv',
    'f_exc, p2=1e-4': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=0_16_elements_shots=1000000.0_09-Apr-2021 (07:54:14.441053).csv',
    'q_exc, p2=1e-4, tcx=10': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=0.0001_tcx=10_16_elements_shots=1000000.0_13-Apr-2021 (07:25:58.710294).csv',
    # 'f_exc, p2=1e-4, tcx=10': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=0.0001_tcx=10_16_elements_shots=1000000.0_13-Apr-2021 (00:16:48.559473).csv',
    # 'q_exc, p2=1e-6': 'results/zhenghao_testing/H4_vqe_q_exc_COBYLA_p=1e-06_tcx=0_16_elements_shots=1000000.0_06-Apr-2021 (22:03:06.233869).csv',
    'f_exc, p2=1e-6': 'results/zhenghao_testing/H4_vqe_eff_f_exc_COBYLA_p=1e-06_tcx=0_16_elements_shots=1000000.0_06-Apr-2021 (12:28:31.699328).csv'
}



num_elem_filename_dict_dict = {
    6: filename_6elem,
    7: filename_dict_7elem,
    8: filename_dict_8elem,
    9: filename_dict_9_elem,
    10: filename_dict_10elem,
    11: filename_dict_11elem,
    12: filename_dict_12elem,
    13: filename_dict_13elem,
    14: filename_dict_14elem,
    16: filename_dict_16elem
}

fig_idx = 0
for num_elem in num_elem_filename_dict_dict.keys():
    filename_dict = num_elem_filename_dict_dict[num_elem]

    plt.figure(fig_idx)
    max_iteration = 0
    for df_label in filename_dict.keys():
        df = pd.read_csv('../../../'+filename_dict[df_label])
        plt.plot(df['iteration'], df['energy']-ground_state, label=df_label)

        if max(df['iteration']) > max_iteration:
            max_iteration = max(df['iteration'])

    plt.hlines(chem_acc, 0, max_iteration,
               label='chem accuracy')
    if fig_idx == 0 or num_elem == 10:
        plt.legend(['q, p2=1e-4', 'f, p2=1e-4', 'q, p2=1e-6', 'f, p2=1e-6',
                    'q, p2=1e-8', 'f, p2=1e-8', 'q, shot noise', 'f, shot noise'])
    plt.xlabel('Iteration')
    plt.ylabel('Energy accuracy (Hartree)')
    plt.yscale('log')
    plt.title('VQE for {} elements with COBYLA optimizer'.format(num_elem))

    plt_filename = 'COBYLA_convergence/{}_COBYLA_convergence.png'.format(num_elem)
    plt.savefig(plt_filename)

    fig_idx += 1
