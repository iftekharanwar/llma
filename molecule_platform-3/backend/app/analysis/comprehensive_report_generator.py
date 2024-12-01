import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen
from datetime import datetime
from ..data_sources.api_clients import DataSourceAggregator
from ..data_sources.fda_client import FDAClient
from ..data_sources.side_effects_client import SideEffectsClient
from ..data_sources.pubchem_client import PubChemClient
from ..data_sources.chembl_client import ChEMBLClient

logger = logging.getLogger(__name__)

class ComprehensiveReportGenerator:
    """Generates comprehensive molecular analysis reports using multiple data sources."""

    def __init__(self, pubchem_client: PubChemClient, chembl_client: ChEMBLClient,
                 side_effects_client: SideEffectsClient, fda_client: FDAClient,
                 risk_assessor: Any):
        self.pubchem_client = pubchem_client
        self.chembl_client = chembl_client
        self.side_effects_client = side_effects_client
        self.fda_client = fda_client
        self.risk_assessor = risk_assessor
        self._sessions = []

    async def __aenter__(self):
        """Initialize all clients."""
        await self.data_aggregator.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup all clients."""
        # Removed data_aggregator cleanup as it's no longer used

    async def generate_comprehensive_report(self, smiles: str, context: str = "medicine") -> Dict[str, Any]:
        """Generate a comprehensive analysis report including similarity, risk, and side effects."""
        try:
            logger.info("Starting comprehensive report generation...")

            # Get similar compounds from PubChem
            similar_compounds = await self.pubchem_client.search_similar_compounds(smiles)
            logger.info(f"Found {len(similar_compounds)} similar compounds from PubChem")

            # Get compound name from similar compounds
            compound_name = similar_compounds[0].get('name', '') if similar_compounds else ''

            # Calculate molecular properties using RDKit
            mol = Chem.MolFromSmiles(smiles)
            mol_properties = self._calculate_molecular_properties(mol)

            # Generate risk assessment
            risk_assessment = await self._generate_risk_assessment(smiles, similar_compounds)
            logger.info("Risk assessment completed")

            # Get side effects data
            side_effects = await self.generate_side_effects_report(compound_name, smiles)
            logger.info("Side effects analysis completed")

            # Calculate confidence metrics
            confidence_metrics = self._calculate_overall_confidence({
                "similarity": len(similar_compounds) > 0,
                "risk": bool(risk_assessment),
                "side_effects": bool(side_effects)
            })

            return {
                "query_molecule": {
                    "name": compound_name,
                    "smiles": smiles,
                    "properties": mol_properties
                },
                "similar_compounds": similar_compounds[:10],  # Limit to top 10 similar compounds
                "risk_assessment": risk_assessment,
                "side_effects": side_effects,
                "confidence_metrics": confidence_metrics,
                "context": context
            }

        except Exception as e:
            logger.error(f"Error generating comprehensive report: {str(e)}")
            raise

    async def generate_similarity_report(self, smiles: str) -> Dict[str, Any]:
        """Generate similarity search report with detailed compound analysis."""
        try:
            # Use data aggregator for similarity search
            async with self.data_aggregator as aggregator:
                similar_compounds = await aggregator.search_similar_compounds(smiles, similarity=0.8)

            # Calculate confidence metrics
            confidence_metrics = {
                'data_completeness': sum(1 for c in similar_compounds if all(
                    k in c for k in ['name', 'smiles', 'similarity', 'properties']
                )) / len(similar_compounds) if similar_compounds else 0,
                'similarity_threshold': 0.8,
                'total_sources': len(set(c.get('source', '') for c in similar_compounds)),
                'property_coverage': self._calculate_property_coverage(similar_compounds)
            }

            return {
                "timestamp": datetime.now().isoformat(),
                "query_molecule": {
                    "smiles": smiles,
                    "properties": self._calculate_molecular_properties(smiles)
                },
                "similar_compounds": similar_compounds,
                "analysis_summary": {
                    "total_compounds_found": len(similar_compounds),
                    "avg_similarity": sum(c.get('similarity', 0) for c in similar_compounds) / len(similar_compounds) if similar_compounds else 0,
                    "data_sources": list(set(c.get('source', '') for c in similar_compounds)),
                    "property_coverage": confidence_metrics['property_coverage']
                },
                "confidence_metrics": confidence_metrics
            }

        except Exception as e:
            logger.error(f"Error generating similarity report: {str(e)}")
            return {"error": str(e)}

    async def _generate_risk_assessment(self, smiles: str, similar_compounds: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate detailed risk assessment using molecular properties and FDA data."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                raise ValueError("Invalid SMILES string")

            # Calculate molecular properties
            properties = self._calculate_molecular_properties(mol)

            # Get FDA adverse event data
            fda_data = await self.fda_client.get_adverse_events(similar_compounds[0].get('name', '')) if similar_compounds else None

            # Analyze molecular risks
            risk_factors = self._assess_risk_factors(mol, properties)

            # Calculate Lipinski compliance
            lipinski_compliance = self._check_lipinski_compliance(properties)

            # Generate property-based alerts
            property_alerts = self._generate_property_alerts(properties, risk_factors)

            # Calculate overall risk score
            risk_score = self._calculate_risk_score(risk_factors, lipinski_compliance)

            # Calculate property coverage for similar compounds
            property_coverage = self._calculate_property_coverage(similar_compounds) if similar_compounds else {}

            # Generate recommendations
            recommendations = self._generate_recommendations(risk_factors)

            return {
                "properties": properties,
                "risk_factors": risk_factors,
                "lipinski_compliance": lipinski_compliance,
                "property_alerts": property_alerts,
                "risk_score": risk_score,
                "property_coverage": property_coverage,
                "recommendations": recommendations,
                "fda_data": fda_data
            }

        except Exception as e:
            logger.error(f"Error in risk assessment: {str(e)}")
            raise

    async def generate_side_effects_report(self, compound_name: str, smiles: str) -> List[Dict[str, Any]]:
        """Generate a report of potential side effects based on similar compounds."""
        try:
            if not compound_name or not smiles:
                return []

            # Get side effects data from both sources
            side_effects_data = await self.side_effects_client.get_side_effects(smiles)
            fda_events = await self.fda_client.get_adverse_events(compound_name)

            # Combine and analyze the data
            combined_effects = []

            # Add side effects from primary source
            if side_effects_data:
                for effect in side_effects_data:
                    combined_effects.append({
                        "effect": effect.get("effect", ""),
                        "severity": effect.get("severity", "unknown"),
                        "frequency": effect.get("frequency", "unknown"),
                        "source": "primary"
                    })

            # Add FDA adverse events
            if fda_events:
                for event in fda_events:
                    combined_effects.append({
                        "effect": event.get("reaction", ""),
                        "severity": event.get("severity", "unknown"),
                        "frequency": event.get("frequency", "unknown"),
                        "source": "FDA"
                    })

            # Sort by severity and frequency
            severity_order = {"high": 3, "moderate": 2, "low": 1, "unknown": 0}
            combined_effects.sort(
                key=lambda x: (severity_order.get(x["severity"], 0), x["frequency"]),
                reverse=True
            )

            return combined_effects

        except Exception as e:
            logger.error(f"Error generating side effects report: {str(e)}")
            return []

    def _analyze_side_effects(self, side_effects_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze aggregated side effects data."""
        if not side_effects_data:
            return {}

        # Aggregate effects across compounds
        aggregated_effects = {}
        total_compounds = len(side_effects_data)

        for compound_data in side_effects_data:
            events = compound_data.get('events', {}).get('events_summary', {})
            for effect, freq in events.get('event_frequency', {}).items():
                if effect not in aggregated_effects:
                    aggregated_effects[effect] = {
                        'count': 0,
                        'weighted_frequency': 0,
                        'compounds': []
                    }
                aggregated_effects[effect]['count'] += 1
                aggregated_effects[effect]['weighted_frequency'] += freq * compound_data['similarity']
                aggregated_effects[effect]['compounds'].append(compound_data['compound'])

        # Sort effects by weighted frequency
        sorted_effects = sorted(
            [
                {
                    'effect': effect,
                    'occurrence_rate': data['count'] / total_compounds,
                    'weighted_frequency': data['weighted_frequency'] / total_compounds,
                    'reported_in': data['compounds']
                }
                for effect, data in aggregated_effects.items()
            ],
            key=lambda x: x['weighted_frequency'],
            reverse=True
        )

        return {
            'common_effects': sorted_effects[:20],
            'total_compounds_analyzed': total_compounds,
            'total_unique_effects': len(aggregated_effects)
        }

    def _calculate_side_effects_confidence(self, side_effects_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate confidence metrics for side effects analysis."""
        if not side_effects_data:
            return {
                'data_completeness': 0.0,
                'similarity_confidence': 0.0,
                'reporting_quality': 0.0
            }

        # Calculate data completeness
        completeness_scores = []
        similarity_scores = []
        quality_scores = []

        for compound_data in side_effects_data:
            events = compound_data.get('events', {})

            # Data completeness
            completeness = 1.0 if events.get('events_summary') else 0.0
            completeness_scores.append(completeness)

            # Similarity confidence
            similarity_scores.append(compound_data['similarity'])

            # Reporting quality from FDA data
            quality = events.get('confidence_metrics', {}).get('reporting_quality', 0.0)
            quality_scores.append(quality)

        return {
            'data_completeness': sum(completeness_scores) / len(completeness_scores),
            'similarity_confidence': sum(similarity_scores) / len(similarity_scores),
            'reporting_quality': sum(quality_scores) / len(quality_scores)
        }

    def _calculate_molecular_properties(self, input_mol: Union[str, Chem.Mol]) -> Dict[str, float]:
        """Calculate comprehensive molecular properties using RDKit.

        Args:
            input_mol: Either a SMILES string or an RDKit Mol object
        """
        if isinstance(input_mol, str):
            mol = Chem.MolFromSmiles(input_mol)
        else:
            mol = input_mol

        if not mol:
            return {}

        return {
            "molecular_weight": Descriptors.ExactMolWt(mol),
            "logp": Crippen.MolLogP(mol),
            "hbd": Descriptors.NumHDonors(mol),
            "hba": Descriptors.NumHAcceptors(mol),
            "tpsa": Descriptors.TPSA(mol),
            "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            "aromatic_rings": Descriptors.NumAromaticRings(mol),
            "heavy_atom_count": Descriptors.HeavyAtomCount(mol),
            "complexity": Descriptors.BertzCT(mol)
        }

    def _assess_risk_factors(self, mol: Chem.Mol, properties: Dict[str, float]) -> List[Dict[str, Any]]:
        """Assess various risk factors based on molecular properties and structure."""
        risk_factors = []

        # LogP assessment (lipophilicity)
        if properties.get('logp', 0) > 3:  # Lowered from 5
            risk_factors.append({
                "factor": "Moderate Lipophilicity",
                "value": properties['logp'],
                "threshold": 3,
                "severity": "medium",
                "description": "May affect absorption and distribution"
            })

        # Molecular weight assessment
        if properties.get('molecular_weight', 0) > 160:  # Lowered from 500 for common drugs
            risk_factors.append({
                "factor": "Molecular Weight",
                "value": properties['molecular_weight'],
                "threshold": 160,
                "severity": "low",
                "description": "Common range for pharmaceutical compounds"
            })

        # TPSA assessment
        if properties.get('tpsa', 0) > 60:  # Lowered from 140
            risk_factors.append({
                "factor": "TPSA Consideration",
                "value": properties['tpsa'],
                "threshold": 60,
                "severity": "low",
                "description": "May affect membrane permeability"
            })

        # Aromatic rings assessment
        if properties.get('aromatic_rings', 0) >= 1:
            risk_factors.append({
                "factor": "Aromatic Structure",
                "value": properties['aromatic_rings'],
                "threshold": 1,
                "severity": "low",
                "description": "Contains aromatic rings - common in drug molecules"
            })

        # Rotatable bonds assessment
        if properties.get('rotatable_bonds', 0) > 3:  # Lowered from 10
            risk_factors.append({
                "factor": "Molecular Flexibility",
                "value": properties['rotatable_bonds'],
                "threshold": 3,
                "severity": "low",
                "description": "Moderate molecular flexibility"
            })

        return risk_factors

    def _calculate_risk_score(self, risk_factors: List[Dict[str, Any]], lipinski_compliance: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall risk score based on risk factors and Lipinski compliance.

        Args:
            risk_factors: List of identified risk factors
            lipinski_compliance: Dictionary containing Lipinski rule compliance data
        """
        severity_weights = {"high": 3, "medium": 2, "low": 1}

        # Calculate base score from risk factors
        risk_score = sum(severity_weights[factor["severity"]] for factor in risk_factors)

        # Adjust score based on Lipinski violations
        risk_score += lipinski_compliance["violation_count"] * 2

        return {
            "score": risk_score,
            "category": "high" if risk_score > 8 else "medium" if risk_score > 4 else "low",
            "lipinski_violations": lipinski_compliance["violation_count"],
            "risk_factor_count": len(risk_factors)
        }

    def _check_lipinski_compliance(self, properties: Dict[str, float]) -> Dict[str, Any]:
        """Check compliance with Lipinski's Rule of Five."""
        violations = []

        if properties.get('molecular_weight', 0) > 500:
            violations.append("Molecular weight > 500")
        if properties.get('logp', 0) > 5:
            violations.append("LogP > 5")
        if properties.get('hbd', 0) > 5:
            violations.append("H-bond donors > 5")
        if properties.get('hba', 0) > 10:
            violations.append("H-bond acceptors > 10")

        return {
            "compliant": len(violations) <= 1,
            "violations": violations,
            "violation_count": len(violations)
        }

    def _generate_property_alerts(self, properties: Dict[str, float], risk_factors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate alerts for concerning property values and risk factors.

        Args:
            properties: Dictionary of molecular properties
            risk_factors: List of identified risk factors from previous analysis
        """
        alerts = []

        # Add property-based alerts
        property_thresholds = {
            "molecular_weight": (500, "high"),
            "logp": (5, "high"),
            "tpsa": (140, "high"),
            "rotatable_bonds": (10, "medium"),
            "complexity": (1000, "medium")
        }

        for prop, (threshold, severity) in property_thresholds.items():
            if prop in properties and properties[prop] > threshold:
                alerts.append({
                    "property": prop,
                    "value": properties[prop],
                    "threshold": threshold,
                    "severity": severity
                })

        # Incorporate risk factors into alerts
        for risk in risk_factors:
            alerts.append({
                "property": risk["factor"],
                "value": risk["value"],
                "threshold": risk["threshold"],
                "severity": risk["severity"]
            })

        return alerts

    def _generate_recommendations(self, risk_factors: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on identified risk factors."""
        recommendations = []

        for factor in risk_factors:
            if factor['factor'] == "High Lipophilicity":
                recommendations.append(
                    "Consider reducing lipophilicity by adding polar groups or reducing aromatic rings"
                )
            elif factor['factor'] == "High Molecular Weight":
                recommendations.append(
                    "Consider simplifying the structure to reduce molecular weight"
                )
            elif factor['factor'] == "High TPSA":
                recommendations.append(
                    "Consider balancing polar surface area to improve membrane permeability"
                )

        return recommendations

    def _calculate_property_coverage(self, compounds: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate coverage statistics for molecular properties across similar compounds.

        Args:
            compounds: List of similar compounds with their properties
        """
        if not compounds:
            return {}

        required_properties = [
            "molecular_weight",
            "logp",
            "tpsa",
            "hbd",
            "hba"
        ]

        coverage = {}
        for prop in required_properties:
            # Handle both direct properties and nested properties
            available = sum(
                1 for c in compounds
                if (isinstance(c, dict) and
                    (c.get(prop) is not None or
                     c.get('properties', {}).get(prop) is not None))
            )
            coverage[prop] = round(available / len(compounds), 2) if compounds else 0.0

        return coverage

    def _calculate_overall_confidence(self, confidence_metrics: Dict[str, bool]) -> Dict[str, float]:
        """Calculate overall confidence score from multiple analysis components."""
        # Weight factors for different components
        weights = {
            'similarity': 0.3,  # ChEMBL and PubChem data reliability
            'risk': 0.3,       # RDKit analysis confidence
            'side_effects': 0.4 # FDA FAERS data confidence
        }

        # Calculate component-specific confidence scores based on boolean success indicators
        component_scores = {
            'similarity': 1.0 if confidence_metrics.get('similarity', False) else 0.0,
            'risk': 1.0 if confidence_metrics.get('risk_assessment', False) else 0.0,
            'side_effects': 1.0 if confidence_metrics.get('side_effects', False) else 0.0
        }

        # Calculate weighted overall confidence
        overall_confidence = sum(
            score * weights[component]
            for component, score in component_scores.items()
        )

        return {
            'overall_confidence': round(overall_confidence, 2),
            'component_confidence': {
                component: round(score, 2)
                for component, score in component_scores.items()
            }
        }
