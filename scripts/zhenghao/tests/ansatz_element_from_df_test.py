import pandas as pd
from src.molecules.molecules import H4
from src.iter_vqe_utils import DataUtils

df_test = pd.read_csv('../../../results/zhenghao_testing/ansatz_element_gradient/H4_q_exc_QiskitSim_p2=1e-06_1th_elem_13-Apr-2021.csv')

q_system = H4(r=1)

ansatz_elements= DataUtils.ansatz_elements_from_data_frame(df_test, q_system)
