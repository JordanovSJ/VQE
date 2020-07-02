import logging
import time
import numpy
import pandas
import ray
import pandas
import datetime

import sys
import ast
sys.path.append('../')

from src.vqe_runner import VQERunner
from src.q_systems import *
from src.ansatz_element_lists import *
from src.backends import QiskitSim
from src.utils import LogUtils
from src.adapt_utils import GradAdaptUtils


def save_data(df_data, molecule, time_stamp, ansatz_element_type=None, frozen_els=None):
    if ansatz_element_type is None:
        ansatz_element_type = 'unspecified'

    try:
        df_data.to_csv('../../results/adapt_vqe_results/{}_{}_{}_{}.csv'.format(molecule.name, ansatz_element_type, frozen_els, time_stamp))
    except FileNotFoundError:
        try:
            df_data.to_csv('results/adapt_vqe_results/{}_{}_{}_{}.csv'.format(molecule.name, ansatz_element_type, frozen_els, time_stamp))
        except FileNotFoundError as fnf:
            print(fnf)


def get_ansatz_from_csv(db):
    ansatz = []
    for i in range(len(init_db)):
        element = init_db.loc[i]['element']
        element_qubits = init_db.loc[i]['element_qubits']
        if element[0] == 'e' and element[4] == 's':
            ansatz.append(EfficientSingleFermiExcitation(*ast.literal_eval(element_qubits)))
        elif element[0] == 'e' and element[4] == 'd':
            ansatz.append(EfficientDoubleFermiExcitation(*ast.literal_eval(element_qubits)))
        elif element[0] == 's' and element[2] == 'q':
            ansatz.append(SingleQubitExcitation(*ast.literal_eval(element_qubits)))
        elif element[0] == 'd' and element[2] == 'q':
            ansatz.append(DoubleQubitExcitation(*ast.literal_eval(element_qubits)))
        else:
            print(element, element_qubits)
            raise Exception('Unrecognized ansatz element.')

    var_pars = list(init_db['var_parameters'])

    return ansatz, var_pars


if __name__ == "__main__":
    # <<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>
    # <<<<<<<<<,simulation parameters>>>>>>>>>>>>>>>>>>>>
    r = 1.316
    # theta = 0.538*numpy.pi # for H20
    frozen_els = {'occupied': [], 'unoccupied': []}
    molecule = BeH2(ham_matrix=True) #(frozen_els=frozen_els)

    ansatz_element_type = 'efficient_fermi_excitation'
    # ansatz_element_type = 'qubit_excitation'
    # ansatz_element_type = 'pauli_word_excitation'

    accuracy = 1e-11  # 1e-3 for chemical accuracy
    # threshold = 1e-14
    max_ansatz_elements = 70

    multithread = True

    init_db = pandas.read_csv("../../results/adapt_vqe_results/BeH2_grad_adapt_EFE_17-Jun-2020.csv")
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    LogUtils.log_cofig()
    logging.info('{}, r={} ,{}'.format(molecule.name, r, ansatz_element_type))

    time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

    # create a vqe runner object
    optimizer = 'BFGS'
    optimizer_options = {'gtol': 1e-07}
    vqe_runner = VQERunner(molecule, backend=QiskitSim, optimizer=optimizer, optimizer_options=optimizer_options)
    hf_energy = molecule.hf_energy
    fci_energy = molecule.fci_energy

    # dataFrame to collect the simulation data
    df_data = pandas.DataFrame(columns=['n', 'E', 'dE', 'error', 'n_iters', 'cnot_count', 'u1_count', 'cnot_depth',
                                        'u1_depth', 'element', 'element_qubits', 'var_parameters'])
    # <<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>?>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # get single excitations
    ansatz_element_pool = SDExcitations(molecule.n_orbitals, molecule.n_electrons,
                                        element_type=ansatz_element_type).get_ansatz_elements()

    message = 'Length of new pool', len(ansatz_element_pool)
    logging.info(message)
    print(message)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    if init_db is None:
        ansatz_elements = []
        var_parameters = []
    else:
        ansatz_elements, var_parameters = get_ansatz_from_csv(init_db)

    iter_count = 0
    current_energy = hf_energy
    previous_energy = 0

    while previous_energy - current_energy >= accuracy and iter_count <= max_ansatz_elements:
        iter_count += 1

        print('New cycle ', iter_count)

        previous_energy = current_energy

        element_to_add, grad = GradAdaptUtils.\
            get_most_significant_ansatz_element(ansatz_element_pool, molecule, vqe_runner.backend,
                                                initial_var_parameters=var_parameters, initial_ansatz=ansatz_elements,
                                                multithread=multithread)

        result = vqe_runner.vqe_run(ansatz_elements=ansatz_elements+[element_to_add],
                                    initial_var_parameters=var_parameters + list(numpy.zeros(element_to_add.n_var_parameters)))


        # calculate the new energy starting from the current var pars and from the zeros. Hopefully avoid local minima
        # set_var_pars = [
        #     var_parameters + [0],
        #     (numpy.random.random(sum([element.n_var_parameters for element in ansatz_elements+[element_to_add]])) - 0.5)/5,
        #     (numpy.random.random(sum([element.n_var_parameters for element in ansatz_elements+[element_to_add]])) - 0.5)/5,
        #     numpy.zeros(sum([element.n_var_parameters for element in ansatz_elements + [element_to_add]]))
        #     ]

        # ray.init(num_cpus=4)
        # ray_ids = [vqe_runner.vqe_run_multithread.remote(self=vqe_runner,
        #                                                  ansatz_elements=ansatz_elements+[element_to_add],
        #                                                  initial_var_parameters=var_pars)
        #            for var_pars in set_var_pars]
        # results = [ray.get(ray_id) for ray_id in ray_ids]
        # result = min(results, key=lambda x: x.fun)
        # ray.shutdown()

        current_energy = result.fun
        delta_e = previous_energy - current_energy

        # get initial guess for the var. params. for the next iteration
        var_parameters = list(result.x)

        if delta_e > 0:
            ansatz_elements.append(element_to_add)

            # write iteration data
            try:
                if element_to_add.order == 1:
                    element_qubits = [element_to_add.qubit_1, element_to_add.qubit_2]
                elif element_to_add.order == 2:
                    element_qubits = [element_to_add.qubit_pair_1, element_to_add.qubit_pair_2]
                else:
                    element_qubits = []
            except AttributeError:
                # this case corresponds to Pauli word excitation
                element_qubits = element_to_add.excitation

            gate_count = QasmUtils.gate_count_from_ansatz_elements(ansatz_elements, molecule.n_orbitals)
            df_data.loc[iter_count] = {'n': iter_count, 'E': current_energy, 'dE': delta_e, 'error': current_energy-fci_energy,
                                       'n_iters': result['n_iters'], 'cnot_count': gate_count['cnot_count'],
                                       'u1_count': gate_count['u1_count'], 'cnot_depth': gate_count['cnot_depth'],
                                       'u1_depth': gate_count['u1_depth'], 'element': element_to_add.element,
                                       'element_qubits': element_qubits, 'var_parameters': 0}
            df_data['var_parameters'] = result.x
            # df_data['var_parameters'] = var_parameters
            # save data
            save_data(df_data, molecule, time_stamp, ansatz_element_type=ansatz_element_type, frozen_els=frozen_els)

            message = 'Add new element to final ansatz {}. Energy {}. Energy change {}, Grad{}, var. parameters: {}' \
                .format(element_to_add.element, current_energy, delta_e, grad, var_parameters)
            logging.info(message)
            print(message)
        else:
            message = 'No contribution to energy decrease. Stop adding elements to the final ansatz'
            logging.info(message)
            print(message)
            break

        print('Added element ', ansatz_elements[-1].element)

    # save data. Not required?
    save_data(df_data, molecule, time_stamp, ansatz_element_type=ansatz_element_type)

    # calculate the VQE for the final ansatz
    vqe_runner_final = VQERunner(molecule, backend=QiskitSim, ansatz_elements=ansatz_elements)
    final_result = vqe_runner_final.vqe_run(ansatz_elements=ansatz_elements)
    t = time.time()

    print(final_result)
    print('Ciao')