import logging
from typing import Dict, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem import Crippen

logger = logging.getLogger(__name__)

class RDKitRiskAssessment:
    """Performs molecular risk assessment using RDKit."""

    def __init__(self):
        self.toxicity_rules = {
            'high_mw': lambda mw: mw > 500,  # Lipinski's Rule
            'high_logp': lambda logp: logp > 5,  # Lipinski's Rule
            'reactive_groups': [
                '[N+](=O)[O-]',  # Nitro groups
                'C(=O)Cl',       # Acid chlorides
                'C(=O)O[CH3]',   # Methyl esters
                '[SH]',          # Thiols
                'C=C=O'          # Ketenes
            ]
        }

    def assess_molecule(self, smiles: str) -> Dict:
        """Perform comprehensive risk assessment of a molecule."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return {'error': 'Invalid SMILES string'}

            # Calculate basic molecular properties
            properties = self._calculate_properties(mol)

            # Assess toxicity risks
            toxicity_risks = self._assess_toxicity_risks(mol, properties)

            # Calculate overall risk scores
            risk_scores = self._calculate_risk_scores(properties, toxicity_risks)

            return {
                'properties': properties,
                'toxicity_risks': toxicity_risks,
                'risk_scores': risk_scores,
                'confidence_metrics': self._calculate_confidence_metrics(properties)
            }

        except Exception as e:
            logger.error(f"Error in risk assessment: {str(e)}")
            return {'error': str(e)}

    def _calculate_properties(self, mol) -> Dict:
        """Calculate molecular properties using RDKit."""
        try:
            return {
                'molecular_weight': Descriptors.ExactMolWt(mol),
                'logp': Crippen.MolLogP(mol),
                'tpsa': Descriptors.TPSA(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'aromatic_rings': len(mol.GetAromaticRings()),
                'reactive_groups_count': self._count_reactive_groups(mol)
            }
        except Exception as e:
            logger.error(f"Error calculating properties: {str(e)}")
            return {}

    def _assess_toxicity_risks(self, mol, properties: Dict) -> Dict:
        """Assess potential toxicity risks based on molecular properties."""
        risks = []

        # Check Lipinski's Rule of 5 violations
        if properties['molecular_weight'] > 500:
            risks.append({
                'type': 'high_molecular_weight',
                'severity': 'moderate',
                'description': 'High molecular weight may affect absorption'
            })

        if properties['logp'] > 5:
            risks.append({
                'type': 'high_lipophilicity',
                'severity': 'high',
                'description': 'High lipophilicity may lead to bioaccumulation'
            })

        # Check for mutagenic risk using SMARTS patterns
        mutagenic_patterns = [
            '[N+](=O)[O-]',  # Nitro groups
            '[N-]=[N+]=[N-]', # Azide groups
            'C(=O)N(O)O',    # N-nitroso groups
        ]

        # Check for carcinogenic risk using SMARTS patterns
        carcinogenic_patterns = [
            'C(=O)Cl',       # Acid chlorides
            '[As]',          # Arsenic compounds
            '[Hg]',          # Mercury compounds
            'C(=S)',         # Thioketones
        ]

        for pattern in mutagenic_patterns + carcinogenic_patterns:
            if Chem.MolFromSmarts(pattern) and mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                risks.append({
                    'type': 'structural_alert',
                    'severity': 'high',
                    'description': 'Contains potentially hazardous structural features'
                })
                break

        if properties['reactive_groups_count'] > 0:
            risks.append({
                'type': 'reactive_groups',
                'severity': 'high',
                'description': 'Contains potentially reactive functional groups'
            })

        return {
            'identified_risks': risks,
            'total_risks': len(risks),
            'risk_severity_distribution': self._calculate_risk_severity_distribution(risks)
        }

    def _count_reactive_groups(self, mol) -> int:
        """Count the number of potentially reactive groups."""
        count = 0
        for smarts in self.toxicity_rules['reactive_groups']:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern:
                count += len(mol.GetSubstructMatches(pattern))
        return count

    def _calculate_risk_scores(self, properties: Dict, toxicity_risks: Dict) -> Dict:
        """Calculate normalized risk scores."""
        try:
            # Base scores on property ranges and identified risks
            property_risk = min(1.0, (
                (properties.get('logp', 0) / 5.0) * 0.4 +
                (properties.get('reactive_groups_count', 0) / 3.0) * 0.4 +
                (min(properties.get('molecular_weight', 0), 1000) / 1000.0) * 0.2
            ))

            toxicity_risk = min(1.0, toxicity_risks['total_risks'] / 3.0)

            return {
                'property_based_risk': round(property_risk * 100, 2),
                'toxicity_risk': round(toxicity_risk * 100, 2),
                'overall_risk': round((property_risk * 0.6 + toxicity_risk * 0.4) * 100, 2)
            }

        except Exception as e:
            logger.error(f"Error calculating risk scores: {str(e)}")
            return {}

    def _calculate_risk_severity_distribution(self, risks: list) -> Dict:
        """Calculate the distribution of risk severities."""
        distribution = {'low': 0, 'moderate': 0, 'high': 0}
        for risk in risks:
            severity = risk.get('severity', 'moderate')
            distribution[severity] = distribution.get(severity, 0) + 1
        return distribution

    def _calculate_confidence_metrics(self, properties: Dict) -> Dict:
        """Calculate confidence metrics for the risk assessment."""
        try:
            # Check completeness of property calculations
            required_properties = [
                'molecular_weight', 'logp', 'tpsa', 'hbd', 'hba',
                'rotatable_bonds', 'aromatic_rings', 'reactive_groups_count'
            ]

            available_properties = sum(
                1 for prop in required_properties
                if properties.get(prop) is not None
            )

            completeness = (available_properties / len(required_properties)) * 100

            return {
                'data_completeness': round(completeness, 2),
                'assessment_confidence': round(completeness * 0.8, 2)  # Slightly reduce confidence
            }

        except Exception as e:
            logger.error(f"Error calculating confidence metrics: {str(e)}")
            return {
                'data_completeness': 0,
                'assessment_confidence': 0
            }
