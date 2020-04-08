from src.vqe_runner import VQERunner
from src.molecules import H2, LiH, HF, BeH2
from src.ansatz_elements import UCCGSD, UCCSD, DoubleExchange, SingleExchange, ESD, EGSD
from src.backends import QiskitSimulation
from src.utils import LogUtils

import logging
import time
import numpy
import pandas
import datetime
import qiskit


# HF UCCSD
# var_parameters = [-8.21930763e-05, -1.18016849e-04, -7.98497167e-05, -1.20558130e-04,
#                      -2.71828500e-03, -5.52887296e-05, -4.24819306e-05, -2.73173964e-03,
#                      -1.24402115e-02, -4.20750089e-05, -3.21141042e-05, -1.24511237e-02,
#                      -4.54559184e-05, -2.55549825e-05, -4.42484180e-05, -2.61868262e-05,
#                      -4.54615087e-05, -2.55497799e-05, -4.42561363e-05, -2.61781939e-05,
#                       2.83893140e-04, -4.37068230e-05, -4.77985128e-04, -8.56411152e-05,
#                      -1.14519633e-04, -5.43056935e-05, -5.43335983e-05, -5.43052205e-05,
#                      -5.43337150e-05 , 4.83408784e-04, -5.06900520e-05,  1.44593276e-06,
#                      -2.71889003e-05, -5.43728740e-05, -5.42843153e-05, -5.43740026e-05,
#                      -5.42840143e-05,  2.41929395e-02, -5.29960230e-05, -2.91286495e-02,
#                      -4.49751624e-05, -5.06522195e-05, -4.49752961e-05, -5.06526232e-05,
#                       2.90414096e-02, -4.22846695e-05, -4.74649165e-05, -4.42913892e-05,
#                      -4.74643183e-05, -4.42908121e-05,  1.35451655e-01, -4.29996748e-05,
#                      -4.35698326e-05, -4.29991005e-05, -4.35698876e-05, -5.36927698e-05,
#                      -5.12717227e-05, -5.36926375e-05, -5.12717375e-05,  1.79177587e-02,
#                      -5.03556900e-05, -5.28622303e-05, -5.28628691e-05, -5.03548319e-05,
#                       1.79208097e-02]

if __name__ == "__main__":

    molecule = LiH
    r = 1.546

    # logging
    LogUtils.log_cofig()

    uccsd = UCCSD(molecule.n_orbitals, molecule.n_electrons)

    ansatz_elements = uccsd.get_ansatz_elements()

    vqe_runner = VQERunner(molecule, backend=QiskitSimulation, ansatz_elements=ansatz_elements,
                           molecule_geometry_params={'distance': r}, print_var_parameters=True)#, optimizer='Nelder-Mead', optimizer_options={'xatol': 2e-3, 'fatol': 1e-3})

    t0 = time.time()
    result = vqe_runner.vqe_run()#ansatz_elements=ansatz_elements, initial_var_parameters=var_parameters)
    t = time.time()

    logging.critical(result)
    print(result)
    print('TIme for running: ', t - t0)

    print('Pizza')


