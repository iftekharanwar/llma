import unittest
from rdkit import Chem
from rdkit.Chem import AllChem
from main import parse_molecule, MoleculeInput

class TestMoleculeHandling(unittest.TestCase):
    def test_smiles_format(self):
        # Test valid SMILES
        smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
        mol_input = MoleculeInput(content=smiles, input_format="smiles")
        mol = parse_molecule(mol_input)
        self.assertIsNotNone(mol)

        # Test invalid SMILES
        with self.assertRaises(Exception):
            invalid_input = MoleculeInput(content="invalid_smiles", input_format="smiles")
            parse_molecule(invalid_input)

    def test_mol_format(self):
        # Test valid MOL
        mol_block = '''
     RDKit          2D

  9  8  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.2990    0.7500    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.5981    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.8971    0.7500    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.1962    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.4952    0.7500    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    7.7942    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    9.0933    0.7500    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   10.3923    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
  3  4  1  0
  4  5  1  0
  5  6  1  0
  6  7  1  0
  7  8  1  0
  8  9  1  0
M  END'''
        mol_input = MoleculeInput(content=mol_block, input_format="mol")
        mol = parse_molecule(mol_input)
        self.assertIsNotNone(mol)

        # Test invalid MOL
        with self.assertRaises(Exception):
            invalid_input = MoleculeInput(content="invalid mol block", input_format="mol")
            parse_molecule(invalid_input)

    def test_pdb_format(self):
        # Test valid PDB (minimal example)
        pdb_block = '''
ATOM      1  N   ALA A   1      27.340  24.430   2.614  1.00  0.00           N
ATOM      2  CA  ALA A   1      26.266  25.413   2.842  1.00  0.00           C
ATOM      3  C   ALA A   1      26.913  26.639   3.531  1.00  0.00           C
ATOM      4  O   ALA A   1      27.886  26.463   4.263  1.00  0.00           O
END'''
        mol_input = MoleculeInput(content=pdb_block, input_format="pdb")
        mol = parse_molecule(mol_input)
        self.assertIsNotNone(mol)

        # Test invalid PDB
        with self.assertRaises(Exception):
            invalid_input = MoleculeInput(content="invalid pdb block", input_format="pdb")
            parse_molecule(invalid_input)

    def test_molecule_input_validation(self):
        # Test valid input
        input_data = {
            "content": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "input_format": "smiles"
        }
        molecule_input = MoleculeInput(**input_data)
        self.assertEqual(molecule_input.input_format, "smiles")

        # Test invalid format
        with self.assertRaises(ValueError):
            MoleculeInput(
                content="CC",
                input_format="invalid"
            )

if __name__ == '__main__':
    unittest.main()
