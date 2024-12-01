from rdkit import Chem
from typing import Dict, Any, List
import logging
import asyncio
from datetime import datetime
from .risk_assessment import RiskAssessor
from app.data_sources.api_clients import DataSourceAggregator

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self):
        self.data_aggregator = DataSourceAggregator()
        self.risk_assessor = RiskAssessor()

    async def generate_comprehensive_report(
        self,
        structure: str,
        context: str = "medicine",
        input_format: str = "smiles"
    ) -> Dict[str, Any]:
        """Generate a comprehensive analysis report for the given molecular structure."""
        try:
            # Initialize report sections
            report = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "input_structure": structure,
                    "context": context,
                    "input_format": input_format,
                    "data_sources": self._get_active_data_sources()
                },
                "similarity_search_results": {},
                "risk_assessment": {},
                "medical_records_analysis": {},
                "side_effects_analysis": {}
            }

            # Convert structure to SMILES if needed
            smiles = structure
            if input_format == "mol":
                mol = Chem.MolFromMolBlock(structure)
                if mol:
                    smiles = Chem.MolToSmiles(mol)
            elif input_format == "pdb":
                # Handle PDB format conversion
                mol = Chem.MolFromPDBBlock(structure)
                if mol:
                    smiles = Chem.MolToSmiles(mol)

            if not smiles:
                raise ValueError("Could not convert input structure to SMILES")

            # Get similar compounds
            similar_compounds = await self.data_aggregator.search_similar_compounds(smiles)

            # Generate similarity search results
            similarity_range = self._get_similarity_range(similar_compounds)
            source_distribution = self._calculate_source_distribution(similar_compounds)

            report["similarity_search_results"] = {
                "similar_compounds": similar_compounds,
                "statistics": {
                    "total_compounds": len(similar_compounds),
                    "similarity_range": similarity_range,
                    "source_distribution": source_distribution
                }
            }

            # Generate risk assessment using RDKit
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                report["risk_assessment"] = self._generate_risk_assessment(mol, similar_compounds)

            # Get medical records data for similar compounds
            medical_records_data = []
            for compound in similar_compounds[:5]:  # Analyze top 5 similar compounds
                if compound.get('name'):
                    records = await self.data_aggregator.get_medical_records(compound['name'])
                    if records:
                        medical_records_data.append(records)

            # Aggregate medical records data
            report["medical_records_analysis"] = self._aggregate_medical_records(medical_records_data)

            # Get side effects data
            side_effects_results = []
            for compound in similar_compounds[:5]:  # Analyze top 5 similar compounds
                if compound.get('name'):
                    side_effects = await self.data_aggregator.get_side_effects(compound['name'])
                    if side_effects:
                        side_effects_results.append(side_effects)

            report["side_effects_analysis"] = self._aggregate_side_effects(side_effects_results)

            # Calculate confidence metrics
            confidence_metrics = self._calculate_overall_confidence(
                similar_compounds,
                report["risk_assessment"],
                report["medical_records_analysis"],
                report["side_effects_analysis"]
            )

            report["metadata"]["confidence_metrics"] = confidence_metrics
            report["metadata"]["recommendations"] = self._generate_confidence_recommendations(
                confidence_metrics
            )

            return report

        except Exception as e:
            logger.error(f"Error generating comprehensive report: {str(e)}")
            raise

    def _get_similarity_range(self, compounds: List[Dict]) -> Dict[str, float]:
        """Calculate similarity score range for compounds."""
        if not compounds:
            return {"min": 0.0, "max": 0.0, "avg": 0.0}

        similarities = [c.get('similarity', 0.0) for c in compounds]
        return {
            "min": round(min(similarities), 2),
            "max": round(max(similarities), 2),
            "avg": round(sum(similarities) / len(similarities), 2)
        }

    def _calculate_source_distribution(self, compounds: List[Dict]) -> Dict[str, int]:
        """Calculate distribution of compound sources."""
        sources = {}
        for compound in compounds:
            source = compound.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        return sources

    def _generate_risk_assessment(self, mol: Chem.Mol, similar_compounds: List[Dict]) -> Dict[str, Any]:
        """Generate risk assessment using RDKit and similar compound data."""
        try:
            # Calculate molecular descriptors
            descriptors = {
                'molecular_weight': Chem.Descriptors.ExactMolWt(mol),
                'logp': Chem.Crippen.MolLogP(mol),
                'tpsa': Chem.Descriptors.TPSA(mol),
                'hbd': Chem.rdMolDescriptors.CalcNumHBD(mol),
                'hba': Chem.rdMolDescriptors.CalcNumHBA(mol),
                'rotatable_bonds': Chem.rdMolDescriptors.CalcNumRotatableBonds(mol),
                'aromatic_rings': Chem.rdMolDescriptors.CalcNumAromaticRings(mol),
                'heavy_atom_count': Chem.rdMolDescriptors.CalcNumHeavyAtoms(mol),
                'complexity': Chem.GraphDescriptors.BertzCT(mol)
            }

            # Calculate drug-likeness scores
            mw_ok = 160 <= descriptors['molecular_weight'] <= 500
            logp_ok = -0.4 <= descriptors['logp'] <= 5.6
            hbd_ok = descriptors['hbd'] <= 5
            hba_ok = descriptors['hba'] <= 10
            tpsa_ok = descriptors['tpsa'] <= 140
            rotatable_ok = descriptors['rotatable_bonds'] <= 10

            drug_likeness = {
                'molecular_weight': {'value': descriptors['molecular_weight'], 'ok': mw_ok},
                'logp': {'value': descriptors['logp'], 'ok': logp_ok},
                'hbd': {'value': descriptors['hbd'], 'ok': hbd_ok},
                'hba': {'value': descriptors['hba'], 'ok': hba_ok},
                'tpsa': {'value': descriptors['tpsa'], 'ok': tpsa_ok},
                'rotatable_bonds': {'value': descriptors['rotatable_bonds'], 'ok': rotatable_ok}
            }

            # Compare with similar compounds
            similar_stats = self._calculate_similar_compound_stats(similar_compounds)

            # Calculate overall risk scores
            property_risks = self._calculate_property_risks(descriptors, similar_stats)

            return {
                'molecular_descriptors': descriptors,
                'drug_likeness': drug_likeness,
                'similar_compound_statistics': similar_stats,
                'risk_scores': property_risks,
                'overall_risk': sum(property_risks.values()) / len(property_risks) if property_risks else 0.0
            }

        except Exception as e:
            logger.error(f"Error generating risk assessment: {str(e)}")
            return {}

    def _calculate_similar_compound_stats(self, compounds: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Calculate statistics from similar compounds."""
        if not compounds:
            return {}

        properties = ['molecular_weight', 'logp', 'tpsa', 'hbd', 'hba', 'rotatable_bonds']
        stats = {}

        for prop in properties:
            values = []
            for comp in compounds:
                val = comp.get('properties', {}).get(prop)
                if val is not None:
                    values.append(float(val))

            if values:
                stats[prop] = {
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values)
                }

        return stats

    def _calculate_property_risks(self, descriptors: Dict[str, float], similar_stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate risk scores for molecular properties."""
        risk_scores = {}

        for prop, value in descriptors.items():
            if prop in similar_stats:
                stats = similar_stats[prop]
                # Calculate how far the value is from the average of similar compounds
                avg = stats['avg']
                range_size = stats['max'] - stats['min']
                if range_size > 0:
                    deviation = abs(value - avg) / range_size
                    risk_scores[prop] = min(deviation, 1.0)
                else:
                    risk_scores[prop] = 0.0

        return risk_scores

    def _calculate_overall_confidence(self, similar_compounds: List[Dict], risk_assessment: Dict,
                                   medical_records: Dict, side_effects: Dict) -> Dict[str, Any]:
        """Calculate overall confidence metrics for the analysis."""
        try:
            similarity_confidence = self._calculate_similarity_confidence(similar_compounds)
            data_completeness = self._calculate_data_completeness({
                'similar_compounds': similar_compounds,
                'risk_analysis': risk_assessment,
                'medical_records': medical_records,
                'side_effects': side_effects
            })
            reliability = self._assess_data_reliability({
                'similar_compounds': similar_compounds,
                'risk_analysis': risk_assessment,
                'medical_records': medical_records,
                'side_effects': side_effects
            })
            coverage = self._calculate_data_coverage({
                'similar_compounds': similar_compounds,
                'medical_records': medical_records,
                'side_effects': side_effects
            })

            component_scores = {
                'similarity': similarity_confidence,
                'completeness': data_completeness,
                'reliability': reliability,
                'coverage': coverage
            }

            overall_confidence = sum(component_scores.values()) / len(component_scores)

            return {
                'overall_confidence': round(overall_confidence, 2),
                'component_scores': component_scores,
                'recommendations': self._generate_confidence_recommendations(component_scores)
            }

        except Exception as e:
            logger.error(f"Error calculating overall confidence: {str(e)}")
            return {
                'overall_confidence': 0.0,
                'error': str(e),
                'component_scores': {},
                'recommendations': ['Error calculating confidence metrics']
            }

    def _calculate_similarity_confidence(self, similar_compounds: List[Dict]) -> float:
        """Calculate confidence based on similarity scores and number of compounds."""
        if not similar_compounds:
            return 0.0

        # Calculate average similarity and adjust for number of compounds
        similarities = [comp.get("similarity", 0) for comp in similar_compounds]
        avg_similarity = sum(similarities) / len(similarities)
        compound_count_factor = min(1.0, len(similar_compounds) / 10)  # Normalize by expected number

        # Weight the confidence score
        similarity_weight = 0.7
        count_weight = 0.3
        confidence = (avg_similarity * similarity_weight) + (compound_count_factor * count_weight)

        return round(min(1.0, confidence), 2)

    def _calculate_data_completeness(self, components: Dict[str, Any]) -> float:
        """Calculate completeness of data across all analysis components."""
        try:
            completeness_scores = {
                'similar_compounds': len(components['similar_compounds']) > 0,
                'risk_analysis': all(key in components['risk_analysis'] for key in ['property_analysis', 'toxicity_assessment']),
                'medical_records': all(key in components['medical_records'] for key in ['usage_statistics', 'outcome_analysis']),
                'side_effects': all(key in components['side_effects'] for key in ['effects_summary', 'severity_distribution'])
            }

            # Calculate weighted completeness score
            weights = {'similar_compounds': 0.3, 'risk_analysis': 0.3, 'medical_records': 0.2, 'side_effects': 0.2}
            completeness = sum(score * weights[component] for component, score in completeness_scores.items())

            return round(completeness, 2)

        except Exception as e:
            logger.error(f"Error calculating data completeness: {str(e)}")
            return 0.0

    def _generate_confidence_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """Generate recommendations based on confidence scores."""
        recommendations = []

        if scores['similarity'] < 0.5:
            recommendations.append("Consider expanding similarity search to include more databases")
        if scores['completeness'] < 0.5:
            recommendations.append("Some analysis components have incomplete data")
        if scores['reliability'] < 0.5:
            recommendations.append("Data reliability concerns detected - verify source data quality")
        if scores['coverage'] < 0.5:
            recommendations.append("Limited data coverage - consider additional data sources")

        if not recommendations:
            recommendations.append("Analysis confidence metrics meet quality thresholds")

        return recommendations

    def _assess_data_reliability(self, components: Dict[str, Any]) -> float:
        """Assess reliability of data across analysis components."""
        try:
            reliability_scores = {
                'similar_compounds': self._assess_compound_data_reliability(components['similar_compounds']),
                'risk_analysis': 0.8,  # RDKit calculations are reliable
                'medical_records': 0.0 if components['medical_records'].get('status') == 'unavailable' else 0.7,
                'side_effects': self._assess_side_effects_reliability(components['side_effects'])
            }

            weights = {'similar_compounds': 0.4, 'risk_analysis': 0.3, 'medical_records': 0.2, 'side_effects': 0.1}
            reliability = sum(score * weights[component] for component, score in reliability_scores.items())

            return round(reliability, 2)

        except Exception as e:
            logger.error(f"Error assessing data reliability: {str(e)}")
            return 0.0

    def _calculate_data_coverage(self, components: Dict[str, Any]) -> float:
        """Calculate coverage of data across analysis components."""
        try:
            coverage_scores = {
                'similar_compounds': self._calculate_compound_coverage(components['similar_compounds']),
                'medical_records': self._calculate_medical_coverage(components['medical_records']),
                'side_effects': self._calculate_side_effects_coverage(components['side_effects'])
            }

            weights = {'similar_compounds': 0.4, 'medical_records': 0.3, 'side_effects': 0.3}
            coverage = sum(score * weights[component] for component, score in coverage_scores.items())

            return round(coverage, 2)

        except Exception as e:
            logger.error(f"Error calculating data coverage: {str(e)}")
            return 0.0

    def _calculate_compound_coverage(self, compounds: List[Dict]) -> float:
        """Calculate coverage score for similar compounds data."""
        if not compounds:
            return 0.0

        required_fields = ['name', 'similarity', 'source', 'properties']
        coverage = sum(
            sum(1 for field in required_fields if field in comp) / len(required_fields)
            for comp in compounds
        ) / len(compounds)

        return coverage

    def _aggregate_medical_records(self, medical_records_data: List[Dict]) -> Dict[str, Any]:
        """Aggregate and analyze medical records data from multiple sources."""
        try:
            if not medical_records_data:
                return {
                    "usage_statistics": {},
                    "outcome_analysis": {},
                    "demographic_distribution": {},
                    "temporal_trends": {},
                    "data_quality": {
                        "total_records": 0,
                        "completeness": 0.0,
                        "time_range": None
                    }
                }

            # Aggregate statistics across all records
            total_records = sum(records.get("data_quality", {}).get("total_records", 0)
                              for records in medical_records_data)

            # Combine usage statistics
            combined_usage = {
                "administration_routes": {},
                "dose_ranges": [],
                "frequency_distribution": {}
            }

            # Combine outcome analysis
            combined_outcomes = {
                "diagnoses_distribution": {},
                "severity_distribution": {},
                "effectiveness_metrics": {}
            }

            # Combine demographic data
            combined_demographics = {
                "age_distribution": {
                    "ranges": {},
                    "average": 0.0
                },
                "gender_distribution": {},
                "population_metrics": {}
            }

            # Process each record set
            for records in medical_records_data:
                # Aggregate usage statistics
                usage_stats = records.get("usage_statistics", {})
                for route, count in usage_stats.get("administration_routes", {}).items():
                    combined_usage["administration_routes"][route] = \
                        combined_usage["administration_routes"].get(route, 0) + count

                # Aggregate outcomes
                outcomes = records.get("outcome_analysis", {})
                for diagnosis, count in outcomes.get("diagnoses_distribution", {}).items():
                    combined_outcomes["diagnoses_distribution"][diagnosis] = \
                        combined_outcomes["diagnoses_distribution"].get(diagnosis, 0) + count

                for severity, count in outcomes.get("severity_distribution", {}).items():
                    combined_outcomes["severity_distribution"][severity] = \
                        combined_outcomes["severity_distribution"].get(severity, 0) + count

                # Aggregate demographics
                demographics = records.get("demographic_distribution", {})
                for gender, count in demographics.get("gender_distribution", {}).items():
                    combined_demographics["gender_distribution"][gender] = \
                        combined_demographics["gender_distribution"].get(gender, 0) + count

            # Calculate average completeness
            avg_completeness = sum(records.get("data_quality", {}).get("completeness", 0)
                                 for records in medical_records_data) / len(medical_records_data) \
                if medical_records_data else 0.0

            return {
                "usage_statistics": combined_usage,
                "outcome_analysis": combined_outcomes,
                "demographic_distribution": combined_demographics,
                "temporal_trends": self._aggregate_temporal_trends(medical_records_data),
                "data_quality": {
                    "total_records": total_records,
                    "completeness": round(avg_completeness, 2),
                    "time_range": self._aggregate_time_ranges(medical_records_data)
                }
            }

        except Exception as e:
            logger.error(f"Error aggregating medical records: {str(e)}")
            return {
                "usage_statistics": {},
                "outcome_analysis": {},
                "demographic_distribution": {},
                "temporal_trends": {},
                "data_quality": {
                    "total_records": 0,
                    "completeness": 0.0,
                    "time_range": None
                },
                "error": str(e)
            }

    def _aggregate_temporal_trends(self, records_data: List[Dict]) -> Dict[str, Any]:
        """Aggregate temporal trends from multiple record sets."""
        combined_trends = {
            "diagnosis_patterns": {},
            "record_distribution": {
                "total_records": 0,
                "unique_diagnoses": set(),
                "data_completeness": 0.0
            }
        }

        for records in records_data:
            trends = records.get("temporal_trends", {})
            patterns = trends.get("diagnosis_patterns", {})

            for diagnosis, data in patterns.items():
                if diagnosis not in combined_trends["diagnosis_patterns"]:
                    combined_trends["diagnosis_patterns"][diagnosis] = {
                        "count": 0,
                        "severity_distribution": {},
                        "routes": {}
                    }

                combined_trends["diagnosis_patterns"][diagnosis]["count"] += data.get("count", 0)

                # Combine severity distributions
                for severity, count in data.get("severity_distribution", {}).items():
                    current = combined_trends["diagnosis_patterns"][diagnosis]["severity_distribution"]
                    current[severity] = current.get(severity, 0) + count

                # Combine route information
                for route, count in data.get("routes", {}).items():
                    current = combined_trends["diagnosis_patterns"][diagnosis]["routes"]
                    current[route] = current.get(route, 0) + count

            # Update unique diagnoses
            combined_trends["record_distribution"]["unique_diagnoses"].update(patterns.keys())

        # Convert unique diagnoses set to count
        combined_trends["record_distribution"]["unique_diagnoses"] = \
            len(combined_trends["record_distribution"]["unique_diagnoses"])

        return combined_trends


    def _aggregate_time_ranges(self, records_data: List[Dict]) -> Dict[str, Any]:
        """Aggregate time ranges from multiple record sets."""
        total_records = 0
        has_time_data = False

        for records in records_data:
            time_range = records.get("data_quality", {}).get("time_range", {})
            if time_range:
                has_time_data = has_time_data or time_range.get("has_time_data", False)
                total_records += time_range.get("record_count", 0)

        return {
            "has_time_data": has_time_data,
            "record_count": total_records
        }

    def _calculate_side_effects_coverage(self, side_effects: Dict) -> float:
        """Calculate coverage score for side effects data."""
        if not side_effects:
            return 0.0

        required_sections = ['effects_summary', 'severity_distribution', 'frequency_analysis', 'confidence_metrics']
        return sum(1 for section in required_sections if section in side_effects) / len(required_sections)

    def _aggregate_side_effects(self, side_effects_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate and analyze side effects data from multiple sources."""
        try:
            if not side_effects_results:
                return {
                    "effects_summary": [],
                    "severity_distribution": {},
                    "frequency_analysis": {},
                    "confidence_metrics": {
                        "data_completeness": 0.0,
                        "source_reliability": 0.0
                    }
                }

            # Aggregate effects across all results
            effects_summary = []
            severity_counts = {"mild": 0, "moderate": 0, "severe": 0}
            frequency_data = {}

            for result in side_effects_results:
                for effect in result.get("effects", []):
                    effect_entry = {
                        "effect": effect["name"],
                        "severity": effect.get("severity", "unknown"),
                        "frequency": effect.get("frequency", "unknown"),
                        "source": effect.get("source", "unknown"),
                        "confidence": effect.get("confidence", 0.0)
                    }
                    effects_summary.append(effect_entry)

                    # Update severity distribution
                    if effect.get("severity") in severity_counts:
                        severity_counts[effect["severity"]] += 1

                    # Update frequency analysis
                    freq = effect.get("frequency", "unknown")
                    frequency_data[freq] = frequency_data.get(freq, 0) + 1

            # Calculate confidence metrics
            data_completeness = sum(1 for effect in effects_summary
                                  if all(k in effect for k in ["severity", "frequency", "confidence"])) / len(effects_summary) if effects_summary else 0.0

            source_reliability = sum(effect.get("confidence", 0.0) for effect in effects_summary) / len(effects_summary) if effects_summary else 0.0

            return {
                "effects_summary": effects_summary,
                "severity_distribution": severity_counts,
                "frequency_analysis": frequency_data,
                "confidence_metrics": {
                    "data_completeness": round(data_completeness, 2),
                    "source_reliability": round(source_reliability, 2)
                }
            }

        except Exception as e:
            logger.error(f"Error aggregating side effects data: {str(e)}")
            return {
                "effects_summary": [],
                "severity_distribution": {},
                "frequency_analysis": {},
                "confidence_metrics": {
                    "data_completeness": 0.0,
                    "source_reliability": 0.0
                },
                "error": str(e)
            }
