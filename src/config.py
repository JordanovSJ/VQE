import scipy
from scipy import optimize
import numpy

# multithreading
multithread = True
ray_options = {'n_cpus': 6}
qiskit_n_threads = 1

# numerical accuracy
floating_point_accuracy = 10e-15
floating_point_accuracy_digits = 15

matrix_size_threshold = 1e7  # in bytes

# use cache. Much faster, requiring much more ram (up to ~ 15GB for LiH, up to ~ 80GB for beH2)
use_cache = True
# <<<<<<<CLASSICAL OPTIMIZER>>>>>>>>>>>>
optimizer = 'L-BFGS-B'
optimizer_tol = 1e-9
# optimizer_bounds_val = numpy.pi/10
# optimizer_bounds = scipy.optimize.Bounds(-optimizer_bounds_val, optimizer_bounds_val)
optimizer_bounds = None

# optimizer_options = {'maxcor': 15, 'ftol': 1e-8, 'gtol': 1e-6, 'eps': 1e-04, 'maxfun': 1000, 'maxiter': 1000,
#                      'iprint': -1, 'maxls': 20}

# use for rescaled d_exc
optimizer_options = {'maxcor': 13, 'ftol': 1e-07, 'gtol': 1e-05, 'eps': 1e-04, 'maxfun': 1500, 'maxiter': 1000,
                     'iprint': -1, 'maxls': 7}

