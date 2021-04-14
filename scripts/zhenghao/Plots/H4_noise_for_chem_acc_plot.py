import pandas as pd
import matplotlib.pyplot as plt

plt.close('all')

df_list = []

df_list.append(pd.read_csv(
    '../../../results/zhenghao_testing/noise_for_chem_acc/H4_noise_for_chem_acc_10-Mar-2021 (13:44:37.867682).csv'))
df_list.append(pd.read_csv(
    '../../../results/zhenghao_testing/noise_for_chem_acc/H4_noise_for_chem_acc_10-Mar-2021 (14:44:24.766056).csv'))
df_list.append(pd.read_csv(
    '../../../results/zhenghao_testing/noise_for_chem_acc/H4_noise_for_chem_acc_10-Mar-2021 (15:02:10.930351).csv'))
df_list.append(pd.read_csv(
    '../../../results/zhenghao_testing/noise_for_chem_acc/H4_noise_for_chem_acc_10-Mar-2021 (15:15:41.150738).csv'))
df_list.append(pd.read_csv(
    '../../../results/zhenghao_testing/noise_for_chem_acc/H4_noise_for_chem_acc_10-Mar-2021 (15:24:37.182879).csv'))

df_shot6_1 = pd.read_csv(
    '../../../results/zhenghao_testing/noise_for_chem_acc/H4_noise_for_chem_acc_10-Mar-2021 (16:37:06.982126).csv')
df_shot6_2 = pd.read_csv(
    '../../../results/zhenghao_testing/noise_for_chem_acc/H4_noise_for_chem_acc_11-Apr-2021 (22:29:23.105541).csv')
df_shot6_3 = pd.read_csv(
    '../../../results/zhenghao_testing/noise_for_chem_acc/H4_noise_for_chem_acc_12-Apr-2021 (10:48:55.541566).csv'
)

df_3 = df_shot6_1.append(df_shot6_2).append(df_shot6_3)
df_list.append(df_3)
num_df = len(df_list)

exact_q_exc = df_list[0]
chem_acc = [1e-3] * 5

plt.figure(1)
plt.plot(df_list[0]['prob_2'], df_list[0]['exact_q_exc'], label="Exact q exc")
plt.plot(df_list[0]['prob_2'], df_list[0]['exact_f_exc'], label="Exact f exc")
plt.plot(df_list[0]['prob_2'], df_list[0]['exp_value_q_exc'], 'X', label="Noisy q exc, t_cx=0")
plt.plot(df_list[0]['prob_2'], df_list[0]['exp_value_f_exc'], 'X', label="Noisy f exc, t_cx=0")
plt.plot(df_3['prob_2'], df_3['exp_value_q_exc'], 'X',
         label = 'Noisy q exc, t_cx=0, 10^6 shots')
plt.plot(df_3['prob_2'], df_3['exp_value_f_exc'], 'X',
         label = 'Noisy f exc, t_cx=0, 10^6 shots')

for df in df_list[1:num_df-1]:
    plt.plot(df['prob_2'], df['exp_value_q_exc'], 'x',
             label="Noisy q exc, t_cx={}".format(df['time_cx'][0]))
    plt.plot(df['prob_2'], df['exp_value_f_exc'], 'x',
             label="Noisy f exc, t_cx={}".format(df['time_cx'][0]))

plt.xscale('symlog', linthreshx=1e-6)
plt.xlabel('prob_2=prob_meas')
plt.ylabel('ham expectation value (Hartree)')
plt.legend()
plt.title('H4 ham expectation value')
plt.show()

plt.figure(2)
plt.plot(df_list[0]['prob_2'], chem_acc, label="Chemical Accuracy")
plt.plot(df_list[0]['prob_2'], abs(df_list[0]['exp_value_q_exc'] - df_list[0]['exact_q_exc']), 'X', label="Noisy q exc, t_cx=0")
plt.plot(df_list[0]['prob_2'], abs(df_list[0]['exp_value_f_exc'] - df_list[0]['exact_f_exc']), 'X', label="Noisy f exc, t_cx=0")

plt.plot(df_3['prob_2'], abs(df_3['exp_value_q_exc'] - df_3['exact_q_exc']), 'X',
         label="Noisy q exc, t_cx=0, 10^6 shots")
plt.plot(df_3['prob_2'], abs(df_3['exp_value_f_exc'] - df_3['exact_f_exc']), 'X',
         label="Noisy f exc, t_cx=0, 10^6 shots")

for df in df_list[1:num_df-1]:
    plt.plot(df['prob_2'], abs(df['exp_value_q_exc'] - df['exact_q_exc']), 'x',
             label="Noisy q exc, t_cx={}".format(df['time_cx'][0]))
    plt.plot(df['prob_2'], abs(df['exp_value_f_exc'] - df['exact_f_exc']), 'x',
             label="Noisy f exc, t_cx={}".format(df['time_cx'][0]))

plt.xscale('symlog', linthreshx=1e-6)
plt.yscale('log')
plt.xlabel('prob_2=prob_meas')
plt.ylabel('Accuracy (Hartree)')
plt.legend()
plt.title('H4 ham expectation value accuracy')
plt.show()
