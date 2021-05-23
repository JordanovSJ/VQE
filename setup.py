from setuptools import setup, find_packages
setup(
    name="VQE",
    version="0.1",
    packages=['src'],
    scripts=['scripts'],
    author='Jordan',
    author_email='jordanovsj@gmail.com',
    install_requires=['ray', 'openfermion', 'openfermionpsi4', 'numpy', 'scipy', 'qiskit']
)
