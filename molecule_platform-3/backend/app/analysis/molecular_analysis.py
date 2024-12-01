import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs
import requests
import json
from typing import Dict, List, Optional, Tuple

class MolecularAnalyzer:
    def __init__(self):
        self.pubchem_base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        self.chembl_base_url = "https://www.ebi.ac.uk/chembl/api/data"

    def parse_structure(self, structure: str, input_format: str) -> Optional[Chem.Mol]:
        """Parse input structure into RDKit molecule object."""
        try:
            if input_format == "smiles":
                mol = Chem.MolFromSmiles(structure)
            elif input_format == "mol":
                mol = Chem.MolFromMolBlock(structure)
            elif input_format == "pdb":
                mol = Chem.MolFromPDBBlock(structure)
            else:
                raise ValueError(f"Unsupported input format: {input_format}")

            if mol is None:
                raise ValueError("Failed to parse molecular structure")

            return mol
        except Exception as e:
            raise ValueError(f"Error parsing structure: {str(e)}")

    async def search_similar_molecules(self, mol: Chem.Mol) -> List[Dict]:
        """Search PubChem and ChEMBL for similar molecules."""
        # Generate Morgan fingerprint for similarity comparison
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)

        similar_molecules = []

        try:
            # Search PubChem
            smiles = Chem.MolToSmiles(mol)
            response = requests.get(
                f"{self.pubchem_base_url}/compound/similarity/smiles/{smiles}/JSON",
                params={"Threshold": 0.8}
            )
            if response.ok:
                data = response.json()
                for compound in data.get("PC_Compounds", [])[:5]:
                    similar_molecules.append({
                        "id": compound["id"]["id"]["cid"],
                        "name": self._get_compound_name(compound),
                        "formula": compound.get("props", {}).get("molecular_formula", ""),
                        "similarity": round(self._calculate_similarity(mol, compound) * 100, 1),
                        "family": self._get_compound_family(compound),
                        "knownUses": self._get_known_uses(compound)
                    })
        except Exception as e:
            print(f"PubChem search error: {str(e)}")

        return similar_molecules

    def analyze_risk(self, mol: Chem.Mol, context: str) -> Dict:
        """Analyze molecular risks based on structure and context."""
        # Calculate basic molecular properties
        mw = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)

        # Context-specific risk analysis
        risk_data = {
            "overall_risk": self._calculate_overall_risk(mol, context),
            "toxicity_risk": self._analyze_toxicity(mol),
            "environmental_impact": self._analyze_environmental_impact(mol),
            "stability_assessment": self._analyze_stability(mol),
            "properties": {
                "molecular_weight": round(mw, 2),
                "logP": round(logp, 2),
                "hydrogen_bond_donors": hbd,
                "hydrogen_bond_acceptors": hba,
                "topological_polar_surface_area": round(tpsa, 2)
            }
        }

        return risk_data

    def get_comparison_data(self, mol: Chem.Mol) -> Dict:
        """Generate synthetic vs natural comparison data."""
        return {
            "efficacy": {
                "synthetic": self._calculate_efficacy(mol, "synthetic"),
                "natural": self._calculate_efficacy(mol, "natural")
            },
            "safety": {
                "synthetic": self._calculate_safety(mol, "synthetic"),
                "natural": self._calculate_safety(mol, "natural")
            },
            "bioavailability": {
                "synthetic": self._calculate_bioavailability(mol, "synthetic"),
                "natural": self._calculate_bioavailability(mol, "natural")
            },
            "environmental": {
                "synthetic": self._calculate_environmental_impact(mol, "synthetic"),
                "natural": self._calculate_environmental_impact(mol, "natural")
            }
        }

    def _calculate_similarity(self, mol1: Chem.Mol, compound: Dict) -> float:
        """Calculate Tanimoto similarity between two molecules."""
        try:
            mol2 = Chem.MolFromSmiles(compound.get("canonical_smiles", ""))
            if mol2 is None:
                return 0.0

            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)

            return DataStructs.TanimotoSimilarity(fp1, fp2)
        except:
            return 0.0

    def _get_compound_name(self, compound: Dict) -> str:
        """Extract compound name from PubChem data."""
        try:
            return compound.get("props", [{}])[0].get("value", {}).get("sval", "Unknown")
        except:
            return "Unknown"

    def _get_compound_family(self, compound: Dict) -> str:
        """Determine compound family based on structure and properties."""
        try:
            mol = Chem.MolFromSmiles(compound.get("canonical_smiles", ""))
            if mol is None:
                return "Unknown"

            # Analyze structural features
            aromatic_rings = Descriptors.NumAromaticRings(mol)
            aliphatic_rings = Descriptors.NumAliphaticRings(mol)
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)

            if aromatic_rings > 0:
                return "Aromatic Compound"
            elif aliphatic_rings > 0:
                return "Alicyclic Compound"
            elif rotatable_bonds > 0:
                return "Flexible Chain Compound"
            else:
                return "Simple Organic Compound"
        except:
            return "Unknown"

    def _get_known_uses(self, compound: Dict) -> List[str]:
        """Get known uses of compound from databases."""
        try:
            # Query PubChem for compound uses
            cid = compound.get("id", {}).get("id", {}).get("cid")
            if cid:
                response = requests.get(
                    f"{self.pubchem_base_url}/compound/cid/{cid}/classification/JSON"
                )
                if response.ok:
                    data = response.json()
                    uses = []
                    for section in data.get("Section", []):
                        if "Information" in section.get("TOCHeading", ""):
                            for info in section.get("Information", []):
                                if "Use" in info.get("Name", ""):
                                    uses.append(info.get("Value", ""))
                    if uses:
                        return uses
            return ["No known uses found"]
        except:
            return ["Error retrieving uses"]

    def _calculate_overall_risk(self, mol: Chem.Mol, context: str) -> str:
        """Calculate overall risk based on molecular properties and context."""
        try:
            # Calculate key risk indicators
            mw = Descriptors.ExactMolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)

            risk_score = 0

            # Molecular weight risk
            if mw > 500: risk_score += 2
            elif mw > 300: risk_score += 1

            # Lipophilicity risk
            if logp > 5: risk_score += 2
            elif logp > 3: risk_score += 1

            # Absorption risk
            if tpsa < 40: risk_score += 2
            elif tpsa < 80: risk_score += 1

            # Context-specific risks
            if context == "medicine":
                if logp > 5 or mw > 500: risk_score += 1
            elif context == "pesticide":
                if logp > 4: risk_score += 2

            # Determine risk level
            if risk_score >= 6:
                return "High"
            elif risk_score >= 3:
                return "Medium"
            else:
                return "Low"
        except:
            return "Unknown"

    def _analyze_toxicity(self, mol: Chem.Mol) -> Dict:
        """Analyze potential toxicity risks."""
        try:
            # Calculate toxicity-related descriptors
            mw = Descriptors.ExactMolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            aromatic_rings = Descriptors.NumAromaticRings(mol)

            risks = []
            risk_level = "Low"

            if logp > 5:
                risks.append("High lipophilicity may lead to bioaccumulation")
                risk_level = "High"
            if tpsa < 40:
                risks.append("Low polar surface area may affect blood-brain barrier penetration")
                risk_level = max(risk_level, "Medium")
            if aromatic_rings > 3:
                risks.append("Multiple aromatic rings may increase toxicity")
                risk_level = max(risk_level, "Medium")

            return {
                "level": risk_level,
                "details": risks if risks else ["No significant toxicity indicators found"]
            }
        except:
            return {"level": "Unknown", "details": ["Error in toxicity analysis"]}

    def _analyze_environmental_impact(self, mol: Chem.Mol) -> Dict:
        """Analyze environmental impact."""
        try:
            logp = Descriptors.MolLogP(mol)
            mw = Descriptors.ExactMolWt(mol)
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)

            impact_level = "Low"
            details = []

            if logp > 4:
                details.append("High lipophilicity suggests potential bioaccumulation")
                impact_level = "High"
            if mw > 500:
                details.append("High molecular weight may affect biodegradation")
                impact_level = max(impact_level, "Medium")
            if rotatable_bonds < 3:
                details.append("Low flexibility may slow biodegradation")
                impact_level = max(impact_level, "Medium")

            return {
                "level": impact_level,
                "details": details if details else ["Low environmental impact expected"]
            }
        except:
            return {"level": "Unknown", "details": ["Error in environmental analysis"]}

    def _analyze_stability(self, mol: Chem.Mol) -> Dict:
        """Analyze molecular stability."""
        try:
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            rings = Descriptors.RingCount(mol)

            stability_level = "High"
            details = []

            if rotatable_bonds > 10:
                stability_level = "Low"
                details.append("High number of rotatable bonds may affect stability")
            elif rotatable_bonds > 5:
                stability_level = "Medium"
                details.append("Moderate molecular flexibility")

            if rings == 0 and rotatable_bonds > 5:
                stability_level = min(stability_level, "Medium")
                details.append("Lack of ring structures with high flexibility")

            return {
                "level": stability_level,
                "details": details if details else ["Stable under normal conditions"]
            }
        except:
            return {"level": "Unknown", "details": ["Error in stability analysis"]}

    def _calculate_efficacy(self, mol: Chem.Mol, type_: str) -> float:
        """Calculate efficacy score."""
        # Implementation would use ML models and empirical data
        return 85.0 if type_ == "synthetic" else 75.0  # Placeholder

    def _calculate_safety(self, mol: Chem.Mol, type_: str) -> float:
        """Calculate safety score."""
        # Implementation would use toxicity predictions and historical data
        return 90.0 if type_ == "synthetic" else 95.0  # Placeholder

    def _calculate_bioavailability(self, mol: Chem.Mol, type_: str) -> float:
        """Calculate bioavailability score."""
        # Implementation would use Lipinski's rules and other predictors
        return 80.0 if type_ == "synthetic" else 70.0  # Placeholder
