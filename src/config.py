import scipy
from scipy import optimize
import numpy

multithread = {'n_cpus': 3}

qiskit_n_threads = 1
qiskit_zero_threshold = 10e-9


# <<<<<<<CLASSICAL OPTIMIZER>>>>>>>>>>>>
optimizer = 'L-BFGS-B'
optimizer_tol = 1e-8
optimizer_bounds_val = numpy.pi/10
optimizer_bounds = scipy.optimize.Bounds(-optimizer_bounds_val, optimizer_bounds_val)
# optimizer_bounds = None

# optimizer_options = {'maxcor': 15, 'ftol': 1e-8, 'gtol': 1e-6, 'eps': 1e-04, 'maxfun': 1000, 'maxiter': 1000,
#                      'iprint': -1, 'maxls': 20}

# use for rescaled d_exc
optimizer_options = {'maxcor': 13, 'ftol': 1e-07, 'gtol': 1e-05, 'eps': 1e-04, 'maxfun': 1500, 'maxiter': 1000,
                     'iprint': -1, 'maxls': 7}

# this settings worked for the adapt_vqe when using the UCCSD elements. However they fail for ESD
#       works well when we have many ESD elements with initial values for the params
#       does now work for few ESD elements
# optimizer_options={'maxcor': 10, 'ftol': 1e-06, 'gtol': 1e-04, 'eps': 1e-04, 'maxfun': 1500, 'maxiter': 1000,
#                    'iprint': -1, 'maxls': 5}

