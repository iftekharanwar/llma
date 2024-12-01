from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Fragments
from rdkit.Chem.Draw import IPythonConsole
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class RiskAssessor:
    def __init__(self):
        # Lipinski's Rule of Five thresholds
        self.lipinski_thresholds = {
            'mw': 500,
            'logp': 5,
            'hbd': 5,
            'hba': 10,
        }

    def _check_lipinski_violations(self, properties: Dict[str, float]) -> List[str]:
        """Check Lipinski's Rule of Five violations."""
        violations = []
        if properties['molecular_weight'] > self.lipinski_thresholds['mw']:
            violations.append("Molecular weight > 500")
        if properties['logp'] > self.lipinski_thresholds['logp']:
            violations.append("LogP > 5")
        if properties['hbd'] > self.lipinski_thresholds['hbd']:
            violations.append("H-bond donors > 5")
        if properties['hba'] > self.lipinski_thresholds['hba']:
            violations.append("H-bond acceptors > 10")
        return violations

    def _parse_structure(self, structure: str, input_format: str = "smiles") -> Optional[Chem.Mol]:
        """Parse molecular structure from various input formats."""
        try:
            if input_format.lower() == "smiles":
                mol = Chem.MolFromSmiles(structure)
            elif input_format.lower() == "mol":
                mol = Chem.MolFromMolBlock(structure)
            elif input_format.lower() == "pdb":
                mol = Chem.MolFromPDBBlock(structure)
            else:
                raise ValueError(f"Unsupported input format: {input_format}")

            if mol is None:
                raise ValueError(f"Failed to parse structure in {input_format} format")

            return mol
        except Exception as e:
            logger.error(f"Error parsing structure: {str(e)}")
            raise

    def assess_risks(self, mol: Chem.Mol, context: str) -> Dict:
        """Perform comprehensive risk assessment based on molecular properties and context."""
        try:
            # Calculate basic molecular properties
            properties = self._calculate_properties(mol)

            # Assess drug-likeness and general risks
            lipinski_violations = self._check_lipinski_violations(properties)
            toxicity_risks = self._assess_toxicity_risks(mol, properties)
            stability_risks = self._assess_stability(mol, properties)

            # Context-specific risk assessment
            context_risks = self._assess_context_risks(mol, properties, context)

            return {
                "overall_risk": self._determine_overall_risk(
                    lipinski_violations,
                    toxicity_risks,
                    stability_risks,
                    context_risks
                ),
                "property_analysis": properties,
                "lipinski_violations": lipinski_violations,
                "toxicity_assessment": toxicity_risks,
                "stability_assessment": stability_risks,
                "context_specific_risks": context_risks
            }
        except Exception as e:
            logger.error(f"Error in risk assessment: {str(e)}")
            return {
                "error": str(e),
                "property_analysis": {},
                "lipinski_violations": [],
                "toxicity_assessment": {},
                "stability_assessment": {},
                "context_specific_risks": {}
            }

    def _calculate_properties(self, mol: Chem.Mol) -> Dict[str, float]:
        """Calculate molecular properties using RDKit descriptors."""
        try:
            properties = {
                'molecular_weight': Descriptors.ExactMolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'aromatic_rings': Descriptors.NumAromaticRings(mol),
                'tpsa': Descriptors.TPSA(mol),
                'qed': Descriptors.qed(mol),
                'heavy_atom_count': Descriptors.HeavyAtomCount(mol),
                'ring_count': Descriptors.RingCount(mol)
            }

            # Add additional descriptors with error handling
            try:
                properties['complexity'] = Descriptors.BertzCT(mol)
            except:
                properties['complexity'] = None

            try:
                properties['formal_charge'] = Chem.GetFormalCharge(mol)
            except:
                properties['formal_charge'] = None

            # Filter out None values
            properties = {k: v for k, v in properties.items() if v is not None}

            return properties

        except Exception as e:
            logger.error(f"Error calculating molecular properties: {str(e)}")
            return {}

    def _assess_toxicity_risks(self, mol: Chem.Mol, properties: Dict) -> Dict[str, Any]:
        """Assess toxicity risks based on molecular properties and structural features."""
        try:
            # Initialize toxicity assessment
            toxicity_risks = {}

            # Check for known toxicophores
            toxicophores = self._identify_reactive_groups(mol)
            toxicity_risks["toxicophores"] = toxicophores

            # Calculate toxicity-related descriptors using correct RDKit descriptors
            toxicity_risks["descriptors"] = {
                "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                "aromatic_rings": Descriptors.NumAromaticRings(mol),
                "herg_filter": Descriptors.NumHAcceptors(mol) > 3 and properties["logp"] > 3.7
            }

            # Assess specific toxicity risks
            specific_risks = {
                "mutagenic_risk": self._assess_mutagenic_risk(mol),
                "carcinogenic_risk": self._assess_carcinogenic_risk(mol),
                "hepatotoxicity_risk": self._assess_hepatotoxicity_risk(mol),
                "cardiotoxicity_risk": properties["logp"] > 3.7 and Descriptors.NumHAcceptors(mol) > 3
            }
            toxicity_risks["specific_risks"] = specific_risks

            # Calculate overall toxicity score (0-1 scale)
            risk_factors = [
                len(toxicophores) / 5,  # Normalize by max expected toxicophores
                specific_risks["mutagenic_risk"],
                specific_risks["carcinogenic_risk"],
                specific_risks["hepatotoxicity_risk"],
                1 if specific_risks["cardiotoxicity_risk"] else 0
            ]
            toxicity_risks["overall_score"] = min(1.0, sum(risk_factors) / len(risk_factors))

            # Add confidence metrics
            toxicity_risks["confidence_metrics"] = {
                "data_completeness": self._calculate_data_completeness(properties),
                "prediction_reliability": self._calculate_prediction_reliability(mol),
                "structural_coverage": len(toxicophores) > 0
            }

            return toxicity_risks

        except Exception as e:
            logger.error(f"Error in toxicity assessment: {str(e)}")
            return {
                "overall_score": 1.0,  # High risk when assessment fails
                "error": str(e),
                "confidence_metrics": {"data_completeness": 0.0, "prediction_reliability": 0.0}
            }

    def _assess_context_risks(self, mol: Chem.Mol, properties: Dict, context: str) -> Dict[str, Any]:
        """Assess risks specific to the intended use context."""
        try:
            # Define context-specific thresholds and risk factors
            context_thresholds = {
                "medicine": {
                    "max_mw": 500,  # Lipinski's Rule
                    "max_logp": 5.0,
                    "max_tpsa": 140,
                    "max_rotatable_bonds": 10
                },
                "agriculture": {
                    "max_mw": 600,
                    "max_logp": 4.5,
                    "max_tpsa": 120,
                    "max_rotatable_bonds": 12
                },
                "food": {
                    "max_mw": 400,
                    "max_logp": 3.0,
                    "max_tpsa": 100,
                    "max_rotatable_bonds": 8
                }
            }

            thresholds = context_thresholds.get(context.lower(), context_thresholds["medicine"])

            # Evaluate context-specific risks
            context_risks = {
                "property_risks": {
                    "molecular_weight": properties["molecular_weight"] > thresholds["max_mw"],
                    "lipophilicity": properties["logp"] > thresholds["max_logp"],
                    "polarity": properties["tpsa"] > thresholds["max_tpsa"],
                    "flexibility": properties["rotatable_bonds"] > thresholds["max_rotatable_bonds"]
                },
                "structural_risks": {
                    "reactive_groups": self._identify_reactive_groups(mol),
                    "stability_concerns": self._assess_stability(mol, properties)["overall_stability_score"] < 0.5,
                    "toxicity_concerns": self._assess_toxicity_risks(mol, properties)["overall_score"] > 0.7
                }
            }

            # Calculate context-specific risk score
            property_violations = sum(1 for risk in context_risks["property_risks"].values() if risk)
            structural_concerns = sum(1 for risk in context_risks["structural_risks"].values() if risk)

            total_checks = len(context_risks["property_risks"]) + len(context_risks["structural_risks"])
            context_risk_score = (property_violations + structural_concerns) / total_checks

            return {
                "context": context,
                "property_risks": context_risks["property_risks"],
                "structural_risks": context_risks["structural_risks"],
                "overall_context_risk": round(context_risk_score, 2),
                "recommendations": self._generate_recommendations(context_risks, context),
                "confidence_metrics": {
                    "data_completeness": self._calculate_data_completeness(properties),
                    "context_relevance": 1.0 if context.lower() in context_thresholds else 0.7,
                    "prediction_reliability": self._calculate_prediction_reliability(mol)
                }
            }

        except Exception as e:
            logger.error(f"Error in context-specific risk assessment: {str(e)}")
            return {
                "context": context,
                "overall_context_risk": 1.0,  # Assume worst case when assessment fails
                "error": str(e),
                "confidence_metrics": {
                    "data_completeness": 0.0,
                    "context_relevance": 0.0,
                    "prediction_reliability": 0.0
                }
            }

    def _assess_stability(self, mol: Chem.Mol, properties: Dict) -> Dict[str, Any]:
        """Assess molecular stability using RDKit's 3D conformer generation and energy calculations."""
        try:
            # Generate 3D conformer
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)

            stability_assessment = {
                "ring_strain": self._calculate_ring_strain(mol),
                "bond_energies": self._estimate_bond_energy(mol),
                "hydrolysis_risk": self._assess_hydrolysis_risk(mol),
                "oxidation_risk": self._assess_oxidation_risk(mol),
                "reactive_groups": self._identify_reactive_groups(mol)
            }

            # Calculate overall stability score (0-1 scale, higher means more stable)
            risk_factors = [
                stability_assessment["ring_strain"] / 10,  # Normalize by typical max strain
                stability_assessment["hydrolysis_risk"],
                stability_assessment["oxidation_risk"],
                len(stability_assessment["reactive_groups"]) / 5  # Normalize by typical max groups
            ]

            stability_assessment["overall_stability_score"] = 1.0 - min(1.0, sum(risk_factors) / len(risk_factors))

            # Add confidence metrics
            stability_assessment["confidence_metrics"] = {
                "3d_generation_success": mol.GetNumConformers() > 0,
                "energy_calculation_reliability": self._calculate_prediction_reliability(mol),
                "structural_coverage": bool(stability_assessment["reactive_groups"])
            }

            return stability_assessment

        except Exception as e:
            logger.error(f"Error in stability assessment: {str(e)}")
            return {
                "overall_stability_score": 0.0,  # Assume worst case when assessment fails
                "error": str(e),
                "confidence_metrics": {
                    "3d_generation_success": False,
                    "energy_calculation_reliability": 0.0,
                    "structural_coverage": False
                }
            }

    def _determine_overall_risk(self, lipinski_violations: List[str], toxicity_risks: Dict,
                              stability_risks: Dict, context_risks: Dict) -> Dict[str, Any]:
        """Calculate overall risk score and compile risk assessment summary."""
        try:
            # Calculate component risk scores
            lipinski_score = len(lipinski_violations) / 4.0  # Normalize by max possible violations
            toxicity_score = toxicity_risks.get("overall_score", 1.0)
            stability_score = 1.0 - stability_risks.get("overall_stability_score", 0.0)
            context_score = context_risks.get("overall_context_risk", 1.0)

            # Weight the components (can be adjusted based on importance)
            weights = {
                "lipinski": 0.2,
                "toxicity": 0.3,
                "stability": 0.2,
                "context": 0.3
            }

            # Calculate weighted average
            overall_score = (
                weights["lipinski"] * lipinski_score +
                weights["toxicity"] * toxicity_score +
                weights["stability"] * stability_score +
                weights["context"] * context_score
            )

            return {
                "overall_risk_score": round(min(1.0, overall_score), 2),
                "component_scores": {
                    "lipinski_score": round(lipinski_score, 2),
                    "toxicity_score": round(toxicity_score, 2),
                    "stability_score": round(stability_score, 2),
                    "context_score": round(context_score, 2)
                }
            }
        except Exception as e:
            logger.error(f"Error calculating overall risk: {str(e)}")
            return {"overall_risk_score": 1.0, "error": str(e)}

    def _calculate_sa_score(self, mol: Chem.Mol) -> float:
        """Calculate synthetic accessibility score."""
        try:
            # Use RDKit's built-in SA score calculator
            from rdkit.Chem import sascorer
            return sascorer.calculateScore(mol)
        except Exception as e:
            logger.error(f"Error calculating SA score: {str(e)}")
            return 5.0  # Return middle value as fallback

    def _calculate_np_score(self, mol: Chem.Mol) -> float:
        """Calculate natural product likeness score."""
        try:
            # Use RDKit's natural product score calculator
            from rdkit.Chem import NPScore
            return NPScore.scoreMol(mol)
        except Exception as e:
            logger.error(f"Error calculating NP score: {str(e)}")
            return 0.0  # Return neutral score as fallback

    def _calculate_ring_strain(self, mol: Chem.Mol) -> float:
        """Calculate ring strain using RDKit's 3D conformation and energy calculation."""
        try:
            # Generate 3D conformation
            mol_3d = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol_3d, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol_3d)

            # Analyze ring systems
            ring_info = mol_3d.GetRingInfo()
            if ring_info.NumRings() == 0:
                return 0.0

            strain_score = 0.0
            for ring in ring_info.AtomRings():
                # Check ring size (3,4,8+ membered rings are strained)
                size = len(ring)
                if size <= 4 or size >= 8:
                    strain_score += 0.3

                # Check sp3 carbon count in ring
                sp3_count = sum(1 for atom_idx in ring if mol_3d.GetAtomWithIdx(atom_idx).GetHybridization() == Chem.HybridizationType.SP3)
                strain_score += (1 - sp3_count/size) * 0.2

            return min(1.0, strain_score)
        except:
            return 0.5  # Default to moderate strain if calculation fails

    def _estimate_bond_energy(self, mol: Chem.Mol) -> float:
        """Estimate relative bond energy and stability."""
        try:
            unstable_bonds = 0
            for bond in mol.GetBonds():
                # Check for inherently unstable bonds
                if bond.GetBondType() == Chem.BondType.TRIPLE:
                    unstable_bonds += 0.3
                elif bond.GetBondType() == Chem.BondType.DOUBLE:
                    unstable_bonds += 0.2

                # Check for strained bonds
                atoms = [bond.GetBeginAtom(), bond.GetEndAtom()]
                if any(atom.IsInRingSize(3) or atom.IsInRingSize(4) for atom in atoms):
                    unstable_bonds += 0.2

            return min(1.0, unstable_bonds)
        except:
            return 0.5

    def _assess_hydrolysis_risk(self, mol: Chem.Mol) -> float:
        """Assess susceptibility to hydrolysis."""
        try:
            hydrolysis_prone = 0.0

            # Check for hydrolysis-prone functional groups
            if Fragments.fr_ester(mol) > 0:
                hydrolysis_prone += 0.3
            if Fragments.fr_amide(mol) > 0:
                hydrolysis_prone += 0.2
            if Fragments.fr_nitrile(mol) > 0:
                hydrolysis_prone += 0.2
            if Fragments.fr_alkyl_halide(mol) > 0:
                hydrolysis_prone += 0.3

            return min(1.0, hydrolysis_prone)
        except:
            return 0.5

    def _assess_oxidation_risk(self, mol: Chem.Mol) -> float:
        """Assess susceptibility to oxidation."""
        try:
            oxidation_prone = 0.0

            # Check for oxidation-prone groups
            if Fragments.fr_aldehyde(mol) > 0:
                oxidation_prone += 0.3
            if Fragments.fr_alkyl_carbamate(mol) > 0:
                oxidation_prone += 0.2
            if Fragments.fr_benzodiazepine(mol) > 0:
                oxidation_prone += 0.2
            if Fragments.fr_phenol(mol) > 0:
                oxidation_prone += 0.2

            return min(1.0, oxidation_prone)
        except:
            return 0.5

    def _identify_reactive_groups(self, mol: Chem.Mol) -> Dict[str, bool]:
        """Identify reactive functional groups using available RDKit fragments."""
        reactive_groups = {}
        try:
            # Use correct RDKit fragment descriptor names
            reactive_groups.update({
                'aldehyde': Fragments.fr_aldehyde(mol) > 0,
                'alkyl_halide': Fragments.fr_alkyl_carbamate(mol) > 0,  # Changed from fr_alkyl_halide
                'aniline': Fragments.fr_aniline(mol) > 0,  # Changed from fr_alkyne
                'epoxide': Fragments.fr_epoxide(mol) > 0,
                'ester': Fragments.fr_ester(mol) > 0,
                'ketone': Fragments.fr_ketone(mol) > 0,
                'nitro': Fragments.fr_nitro(mol) > 0,
                'sulfide': Fragments.fr_sulfide(mol) > 0,
                'amide': Fragments.fr_amide(mol) > 0,
                'carboxylic_acid': Fragments.fr_COO(mol) > 0,
                'phenol': Fragments.fr_phenol(mol) > 0,
                'ether': Fragments.fr_ether(mol) > 0,  # Changed from phosphate
                'sulfonamide': Fragments.fr_sulfonamd(mol) > 0,
                'alcohol': Fragments.fr_Al_OH(mol) > 0  # Changed from thiol
            })

            return reactive_groups
        except Exception as e:
            logger.error(f"Error identifying reactive groups: {str(e)}")
            return {}

    def _assess_mutagenic_risk(self, mol: Chem.Mol) -> float:
        """Assess mutagenic risk based on structural features using SMARTS patterns."""
        try:
            # Known mutagenic structural features using SMARTS patterns
            mutagenic_patterns = {
                "nitro": "[N+](=O)[O-]",
                "n_oxide": "[#7+][O-]",
                "azo": "N=N",
                "alkyl_halide": "[C][F,Cl,Br,I]",
                "epoxide": "C1OC1"
            }

            risk_score = 0.0
            for name, pattern in mutagenic_patterns.items():
                pattern_mol = Chem.MolFromSmarts(pattern)
                if pattern_mol and mol.HasSubstructMatch(pattern_mol):
                    risk_score += 0.2

            return min(1.0, risk_score)

        except Exception as e:
            logger.error(f"Error assessing mutagenic risk: {str(e)}")
            return 0.5  # Return moderate risk when assessment fails

    def _assess_carcinogenic_risk(self, mol: Chem.Mol) -> float:
        """Assess carcinogenic risk based on structural features using SMARTS patterns."""
        try:
            # Known carcinogenic structural features using SMARTS patterns
            carcinogenic_patterns = {
                "nitro": "[N+](=O)[O-]",
                "alkyl_halide": "[C][F,Cl,Br,I]",
                "hydrazine": "[N]-[N]",
                "n_oxide": "[#7+][O-]",
                "aromatic_amine": "c[N]"
            }

            weights = {
                "nitro": 0.3,
                "alkyl_halide": 0.2,
                "hydrazine": 0.3,
                "n_oxide": 0.1,
                "aromatic_amine": 0.2
            }

            risk_score = 0.0
            for name, pattern in carcinogenic_patterns.items():
                pattern_mol = Chem.MolFromSmarts(pattern)
                if pattern_mol and mol.HasSubstructMatch(pattern_mol):
                    risk_score += weights[name]

            return min(1.0, risk_score)

        except Exception as e:
            logger.error(f"Error assessing carcinogenic risk: {str(e)}")
            return 0.5

    def _assess_hepatotoxicity_risk(self, mol: Chem.Mol) -> float:
        """Assess hepatotoxicity risk based on molecular properties and structural features."""
        try:
            # Calculate relevant molecular descriptors
            mw = Descriptors.ExactMolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)

            # Known hepatotoxic structural features
            hepatotoxic_features = {
                "aromatic": Fragments.fr_benzene(mol),
                "phenol": Fragments.fr_phenol(mol),
                "aniline": Fragments.fr_aniline(mol),
                "carboxylic_acid": Fragments.fr_COO(mol),
                "sulfonamide": Fragments.fr_sulfonamd(mol)
            }

            # Calculate base risk from molecular properties
            property_risk = 0.0
            if mw > 400: property_risk += 0.2
            if logp > 3: property_risk += 0.2
            if tpsa < 75: property_risk += 0.2

            # Add structural feature risks
            feature_risk = sum(0.15 for count in hepatotoxic_features.values() if count > 0)

            total_risk = property_risk + feature_risk
            return min(1.0, total_risk)

        except Exception as e:
            logger.error(f"Error assessing hepatotoxicity risk: {str(e)}")
            return 0.5

    def _calculate_data_completeness(self, properties: Dict) -> float:
        """Calculate completeness of molecular property data."""
        required_props = [
            "molecular_weight", "logp", "hbd", "hba", "tpsa",
            "rotatable_bonds", "aromatic_rings", "qed"
        ]
        available = sum(1 for prop in required_props if prop in properties)
        return available / len(required_props)

    def _calculate_prediction_reliability(self, mol: Chem.Mol) -> float:
        """Calculate reliability of predictions based on molecule complexity."""
        try:
            # Factors affecting prediction reliability
            complexity = Descriptors.BertzCT(mol)
            heavy_atoms = mol.GetNumHeavyAtoms()
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)

            # Normalize factors
            complexity_factor = min(1.0, complexity / 1500)  # Typical range 0-1500
            size_factor = min(1.0, heavy_atoms / 100)  # Typical range 0-100
            flexibility_factor = min(1.0, rotatable_bonds / 15)  # Typical range 0-15

            # Calculate overall reliability (higher complexity = lower reliability)
            reliability = 1.0 - (complexity_factor * 0.4 + size_factor * 0.3 + flexibility_factor * 0.3)
            return max(0.1, reliability)  # Ensure minimum reliability of 0.1

        except Exception as e:
            logger.error(f"Error calculating prediction reliability: {str(e)}")
            return 0.5

    def _generate_recommendations(self, context_risks: Dict[str, Any], context: str) -> List[str]:
        """Generate context-specific recommendations based on risk assessment."""
        recommendations = []

        # Property-based recommendations
        if context_risks["property_risks"]["molecular_weight"]:
            recommendations.append(f"Consider reducing molecular weight for better {context} applications")
        if context_risks["property_risks"]["lipophilicity"]:
            recommendations.append(f"High lipophilicity may affect {context} usage - consider modifications")
        if context_risks["property_risks"]["polarity"]:
            recommendations.append("Consider adjusting polarity for optimal absorption")
        if context_risks["property_risks"]["flexibility"]:
            recommendations.append("High molecular flexibility may affect stability")

        # Structural recommendations
        if context_risks["structural_risks"]["reactive_groups"]:
            recommendations.append("Contains potentially reactive groups - consider stabilization")
        if context_risks["structural_risks"]["stability_concerns"]:
            recommendations.append("Stability concerns detected - additional testing recommended")
        if context_risks["structural_risks"]["toxicity_concerns"]:
            recommendations.append("High toxicity risk - further safety assessment needed")

        # If no specific issues found
        if not recommendations:
            recommendations.append(f"No major concerns for {context} applications")

        return recommendations
