from setuptools import setup, find_packages
setup(
    name="VQE",
    version="0.1",
    packages=['src'],
    scripts=['adapt_vqe'],
    author='Jordan',
    author_email='jordanovsj@gmail.com',
    install_requires=['ray', 'openfermion', 'openfermionpsi4', 'numpy', 'scipy', 'qiskit']
)
