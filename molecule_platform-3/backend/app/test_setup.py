import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

def test_environment():
    # Test SMILES parsing
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        print("❌ Failed to parse SMILES")
        return False

    # Test 3D structure generation
    try:
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        print("✅ 3D structure generation successful")
    except Exception as e:
        print(f"❌ Failed to generate 3D structure: {e}")
        return False

    # Test 2D depiction
    try:
        img = Draw.MolToImage(mol)
        print("✅ 2D depiction successful")
    except Exception as e:
        print(f"❌ Failed to generate 2D depiction: {e}")
        return False

    print("✅ All environment tests passed")
    return True

if __name__ == "__main__":
    print("Testing molecular analysis environment...")
    test_environment()
