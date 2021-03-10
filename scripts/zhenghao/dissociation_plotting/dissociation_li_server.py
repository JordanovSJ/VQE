import pandas as pd
import numpy as np
import sys
sys.path.append('../../')

import matplotlib.pyplot as plt
from src.molecules.molecules import *
from src.utils import *
from scripts.zhenghao.dissociation_plotting.iqeb_fun_li import iqeb_litest

r_list = [1, 1.5, 2, 2.5, 3, 3.5]

molecule_type = H2
ansatz_element_type = 'eff_f_exc'

data_frame = pd.DataFrame(columns=['r', 'fci', 'hf', 'iqeb'])
time_stamp =datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

idx=0
for r in r_list:
    molecule = molecule_type(r)
    vqe_result = iqeb_litest(r, molecule, ansatz_element_type, 1e-5)
    vqe_energy = vqe_result.fun

    data_frame.loc[idx] = {'r': r, 'fci': molecule.fci_energy, 'hf': molecule.hf_energy,
                           'iqeb': vqe_energy}
    data_frame.to_csv('../../results/zhenghao_testing/'
                      '{}_{}_dissociation_curve_{}.csv'.format(molecule.name,
                                                               ansatz_element_type, time_stamp))

    idx += 1
