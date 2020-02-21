vqe_params = {'chemical_accuracy': 1e-3, 'max_n_iterations': 2000}

multithread = {'n_cpus': 6}

adaptive_ansatz_params = {'energy_threshold': 1e-6, 'max_ansatz_elements': 10}

# <<<<<<<CLASSICAL OPTIMIZER>>>>>>>>>>>>
optimizer = 'L-BFGS-B'
optimizer_tol = 1e-5
optimizer_options = {'maxcor': 10, 'ftol': 1e-07, 'gtol': 1e-07, 'eps': 1e-04, 'maxfun': 1000, 'maxiter': 1000,
                     'iprint': -1, 'maxls': 10}
# # the comment code below is the most optimal set up for the optimizer so far
#  method='L-BFGS-B',
#  options={'maxcor': 10, 'ftol': 1e-07, 'gtol': 1e-07,
#           'eps': 1e-04, 'maxfun': 1000, 'maxiter': max_n_iterations,
#           'iprint': -1, 'maxls': 10}, tol=1e-5)
