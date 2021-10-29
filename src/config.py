# multithreading
multithread = True
ray_options = {'n_cpus': 3, 'object_store_memory': None}
multithread_chunk_size = 1000  # number of objects (e.g. commutators) to simultaneously calculate with ray
qiskit_n_threads = 1

# numerical accuracy
floating_point_accuracy = 10e-15
floating_point_accuracy_digits = 15
matrix_size_threshold = 1e7  # in bytes

# optimizer options
default_optimizer = 'BFGS'
default_optimizer_options = {'gtol': 10e-8}
optimizer_tol = 1e-10
optimizer_bounds = None

