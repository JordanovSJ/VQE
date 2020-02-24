import scipy
import numpy

vqe_params = {'chemical_accuracy': 1e-3, 'max_n_iterations': 2000}

multithread = {'n_cpus': 6}

adaptive_ansatz_params = {'energy_threshold': 1e-6, 'max_ansatz_elements': 10}

# <<<<<<<CLASSICAL OPTIMIZER>>>>>>>>>>>>
optimizer = 'L-BFGS-B'
optimizer_tol = 1e-6
optimizer_bounds = scipy.optimize.Bounds(0, 2*numpy.pi)
# optimizer_bounds = None
optimizer_options = {'maxcor': 15, 'ftol': 1e-8, 'gtol': 1e-8, 'eps': 1e-02, 'maxfun': 1000, 'maxiter': 1000,
                     'iprint': -1, 'maxls': 15}
# # the comment code below is the most optimal set up for the optimizer so far

