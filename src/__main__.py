from openfermion.hamiltonians import MolecularData

if __name__ == "__main__":

    from examples.h2 import molecule

    basis = 'sto-3g'
    h2 = MolecularData(molecule.geometry(0.74), basis, molecule.multiplicity, molecule.charge)

    print(h2)

    print('Pizza')
