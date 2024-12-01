from typing import Dict, Any, Optional, List
import logging
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from .risk_assessment import RiskAssessor
from ..data_sources.api_clients import DataSourceAggregator
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

class ComprehensiveReportGenerator:
    """Generates comprehensive molecular analysis reports using real data from multiple sources."""

    def __init__(self):
        self.data_aggregator = DataSourceAggregator()
        self.risk_assessor = RiskAssessor()
        self._active_data_sources = []

    async def __aenter__(self):
        """Initialize data sources."""
        await self.data_aggregator.__aenter__()
        self._active_data_sources = await self.data_aggregator.get_active_sources()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup data sources."""
        await self.data_aggregator.__aexit__(exc_type, exc_val, exc_tb)

    async def generate_similarity_report(self, structure: str) -> Dict[str, Any]:
        """Generate similarity search report for the given structure."""
        try:
            results = await self.data_aggregator.search_similar_compounds(structure)
            if not results.get("similar_compounds"):
                logger.warning("No similar compounds found")
            return {
                "similar_compounds": results.get("similar_compounds", []),
                "total_compounds_found": len(results.get("similar_compounds", [])),
                "data_sources": results.get("data_sources", []),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return {"similar_compounds": [], "total_compounds_found": 0, "data_sources": [], "error": str(e)}

    async def generate_risk_assessment(self, structure: str, context: str = "medicine") -> Dict[str, Any]:
        """Generate risk assessment report."""
        try:
            mol = self.risk_assessor._parse_structure(structure)
            if not mol:
                raise ValueError("Failed to parse structure")
            risk_data = self.risk_assessor.assess_risks(mol, context)
            return {
                "property_analysis": risk_data.get("property_analysis", {}),
                "toxicity_assessment": risk_data.get("toxicity_assessment", {}),
                "recommendations": risk_data.get("recommendations", []),
                "overall_risk_score": risk_data.get("overall_risk", 1.0),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in risk assessment: {str(e)}")
            return {
                "property_analysis": {},
                "toxicity_assessment": {},
                "recommendations": [],
                "overall_risk_score": 1.0,
                "error": str(e)
            }

    async def generate_side_effects_report(self, structure: str, compound_name: str) -> Dict[str, Any]:
        """Generate side effects report."""
        try:
            side_effects = await self.data_aggregator.get_side_effects(compound_name)
            return {
                "effects": [
                    {
                        "effect": effect.get("name"),
                        "frequency": effect.get("frequency"),
                        "severity": effect.get("severity"),
                        "source": effect.get("source")
                    }
                    for effect in side_effects.get("effects", [])
                ],
                "total_effects": len(side_effects.get("effects", [])),
                "confidence_metrics": side_effects.get("confidence", {}),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in side effects analysis: {str(e)}")
            return {
                "effects": [],
                "total_effects": 0,
                "confidence_metrics": {},
                "error": str(e)
            }

    async def generate_medical_records_report(self, compound_name: str) -> Dict[str, Any]:
        """Generate medical records report for a compound."""
        try:
            medical_data = await self.data_aggregator.get_medical_records(compound_name)

            if not medical_data:
                return {
                    "usage_stats": {},
                    "outcomes": [],
                    "confidence_metrics": {"data_completeness": 0.0},
                    "error": "No medical records data available"
                }

            return {
                "usage_stats": {
                    "total_prescriptions": medical_data.get("total_prescriptions", 0),
                    "average_duration": medical_data.get("average_duration", "N/A"),
                    "common_dosages": medical_data.get("common_dosages", []),
                    "administration_routes": medical_data.get("administration_routes", {})
                },
                "outcomes": [
                    {
                        "outcome": outcome.get("description"),
                        "frequency": outcome.get("frequency"),
                        "confidence": outcome.get("confidence", 0.0)
                    }
                    for outcome in medical_data.get("outcomes", [])
                ],
                "confidence_metrics": {
                    "data_completeness": medical_data.get("confidence", {}).get("completeness", 0.0),
                    "source_reliability": medical_data.get("confidence", {}).get("reliability", 0.0)
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in medical records analysis: {str(e)}")
            return {
                "usage_stats": {},
                "outcomes": [],
                "confidence_metrics": {"data_completeness": 0.0},
                "error": str(e)
            }

    async def generate_report(self, structure: str, context: str = "medicine", input_format: str = "smiles") -> Dict[str, Any]:
        """Generate a comprehensive report for the given molecular structure."""
        try:
            # Get similarity search results
            similarity_results = await self.generate_similarity_report(structure)

            # Get risk assessment
            risk_results = await self.generate_risk_assessment(structure)  # Changed from _generate_risk_assessment

            # Get side effects data if we have a compound name from similarity search
            side_effects_results = {}
            if similarity_results.get("similar_compounds"):
                compound_name = similarity_results["similar_compounds"][0].get("name", "")
                if compound_name:
                    side_effects_results = await self.generate_side_effects_report(structure, compound_name)

            return {
                "similarity_analysis": similarity_results,
                "risk_assessment": risk_results,
                "side_effects": side_effects_results,
                "timestamp": datetime.now().isoformat(),
                "data_sources": self.data_aggregator.get_active_sources(),
                "analysis_context": context
            }

        except Exception as e:
            logger.error(f"Error generating comprehensive report: {str(e)}")
            return self._generate_empty_report(structure, input_format, context)

    def _generate_empty_report(self, structure: str, input_format: str, context: str) -> Dict[str, Any]:
        """Generate an empty report template when analysis fails."""
        return {
            "timestamp": datetime.now().isoformat(),
            "input_structure": {
                "format": input_format,
                "structure": structure,
                "parsed_successfully": False
            },
            "similarity_search_results": {
                "similar_compounds": [],
                "total_compounds_found": 0,
                "data_sources": [],
                "search_parameters": {
                    "min_similarity": 0.7,
                    "max_results": 10
                }
            },
            "risk_assessment": {},
            "side_effects_analysis": {},
            "medical_records_analysis": {},
            "analysis_metadata": {
                "context": context,
                "error": "Failed to generate comprehensive report",
                "confidence_metrics": {
                    "overall_confidence": 0.0,
                    "data_completeness": 0.0,
                    "source_reliability": 0.0
                }
            }
        }

    def _aggregate_medical_records(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate medical records data from multiple sources."""
        try:
            aggregated_data = {
                "usage_statistics": {},
                "demographic_distribution": {},
                "outcome_analysis": {},
                "temporal_trends": {},
                "data_quality": {
                    "total_records": 0,
                    "completeness": 0.0,
                    "sources_count": 0
                }
            }

            valid_results = [r for r in results if isinstance(r, dict) and not isinstance(r, Exception)]

            if not valid_results:
                return aggregated_data

            # Combine usage statistics
            for result in valid_results:
                usage_stats = result.get("usage_statistics", {})
                for route, count in usage_stats.get("administration_routes", {}).items():
                    if route not in aggregated_data["usage_statistics"]:
                        aggregated_data["usage_statistics"][route] = 0
                    aggregated_data["usage_statistics"][route] += count

            # Aggregate demographic data
            demo_data = [r.get("demographic_distribution", {}) for r in valid_results]
            if demo_data:
                age_ranges = [d.get("age_distribution", {}).get("range", {}) for d in demo_data if d]
                gender_dists = [d.get("gender_distribution", {}) for d in demo_data if d]

                aggregated_data["demographic_distribution"] = {
                    "age_distribution": {
                        "range": {
                            "min": min([r.get("min") for r in age_ranges if r.get("min") is not None], default=None),
                            "max": max([r.get("max") for r in age_ranges if r.get("max") is not None], default=None)
                        }
                    },
                    "gender_distribution": self._merge_dictionaries([g for g in gender_dists if g])
                }

            # Combine outcome analysis
            outcome_data = [r.get("outcome_analysis", {}) for r in valid_results]
            if outcome_data:
                aggregated_data["outcome_analysis"] = {
                    "diagnoses_distribution": self._merge_dictionaries(
                        [o.get("diagnoses_distribution", {}) for o in outcome_data]
                    ),
                    "severity_distribution": self._merge_dictionaries(
                        [o.get("severity_distribution", {}) for o in outcome_data]
                    )
                }

            # Update data quality metrics
            total_records = sum(r.get("data_quality", {}).get("total_records", 0) for r in valid_results)
            avg_completeness = sum(r.get("data_quality", {}).get("completeness", 0) for r in valid_results) / len(valid_results) if valid_results else 0

            aggregated_data["data_quality"] = {
                "total_records": total_records,
                "completeness": round(avg_completeness, 2),
                "sources_count": len(valid_results)
            }

            return aggregated_data

        except Exception as e:
            logger.error(f"Error aggregating medical records: {str(e)}")
            return {
                "usage_statistics": {},
                "demographic_distribution": {},
                "outcome_analysis": {},
                "temporal_trends": {},
                "data_quality": {"error": str(e)}
            }

    def _generate_confidence_recommendations(self, component_scores: Dict[str, float]) -> List[str]:
        """Generate recommendations based on confidence scores."""
        recommendations = []

        # Check similarity search confidence
        if component_scores["similarity"] < 0.5:
            recommendations.append(
                "Consider expanding similarity search to additional databases or adjusting similarity thresholds"
            )

        # Check risk assessment confidence
        if component_scores["risk_assessment"] < 0.5:
            recommendations.append(
                "Additional molecular property analysis recommended due to limited prediction reliability"
            )

        # Check side effects confidence
        if component_scores["side_effects"] < 0.5:
            recommendations.append(
                "Limited side effects data available - consider cross-referencing with additional medical databases"
            )

        # Check medical records confidence
        if component_scores["medical_records"] < 0.5:
            recommendations.append(
                "Limited medical records data - findings should be validated with additional clinical data sources"
            )

        # Add general recommendations based on overall pattern
        low_confidence_components = [
            component for component, score in component_scores.items()
            if score < 0.7
        ]

        if len(low_confidence_components) >= 3:
            recommendations.append(
                "Multiple low-confidence components detected - consider additional validation studies"
            )

        # If all components have good confidence
        if all(score >= 0.7 for score in component_scores.values()):
            recommendations.append(
                "All analysis components show good confidence levels - results can be considered reliable"
            )

        return recommendations

    def _calculate_comprehensive_confidence(self, similar_compounds: list, risk_assessment: Dict,
                                         side_effects_data: Dict, medical_records_data: Dict) -> Dict[str, Any]:
        """Calculate comprehensive confidence metrics for the entire analysis."""
        try:
            # Calculate component-specific confidence scores
            similarity_confidence = self._calculate_similarity_confidence(similar_compounds)
            risk_confidence = self._calculate_risk_confidence(risk_assessment)
            effects_confidence = self._calculate_effects_confidence(side_effects_data)
            records_confidence = self._calculate_records_confidence(medical_records_data)

            # Calculate weighted overall confidence
            weights = {
                "similarity": 0.3,
                "risk_assessment": 0.3,
                "side_effects": 0.2,
                "medical_records": 0.2
            }

            component_scores = {
                "similarity": similarity_confidence,
                "risk_assessment": risk_confidence,
                "side_effects": effects_confidence,
                "medical_records": records_confidence
            }

            overall_confidence = sum(
                score * weights[component]
                for component, score in component_scores.items()
            )

            return {
                "overall_confidence": round(overall_confidence, 2),
                "component_scores": component_scores,
                "data_quality": {
                    "completeness": self._calculate_report_completeness(
                        similar_compounds, risk_assessment
                    ),
                    "reliability": round(
                        (similarity_confidence + risk_confidence) / 2, 2
                    )
                },
                "recommendations": self._generate_confidence_recommendations(component_scores)
            }

        except Exception as e:
            logger.error(f"Error calculating comprehensive confidence: {str(e)}")
            return {
                "overall_confidence": 0.0,
                "component_scores": {},
                "data_quality": {"error": str(e)},
                "recommendations": ["Error calculating confidence metrics"]
            }

    def _calculate_similarity_confidence(self, similar_compounds: list) -> float:
        """Calculate confidence score for similarity search results."""
        if not similar_compounds:
            return 0.0

        # Consider number and quality of matches
        num_compounds = len(similar_compounds)
        avg_similarity = sum(
            compound.get("similarity", 0)
            for compound in similar_compounds
        ) / num_compounds if num_compounds > 0 else 0

        # Weight both factors
        quantity_score = min(num_compounds / 10, 1.0)  # Normalize to max 10 compounds
        quality_score = avg_similarity

        return round((quantity_score * 0.4 + quality_score * 0.6), 2)

    def _calculate_risk_confidence(self, risk_assessment: Dict) -> float:
        """Calculate confidence score for risk assessment."""
        if not risk_assessment:
            return 0.0

        confidence_metrics = risk_assessment.get("confidence_metrics", {})
        return round(confidence_metrics.get("prediction_reliability", 0.0), 2)

    def _calculate_effects_confidence(self, side_effects_data: Dict) -> float:
        """Calculate confidence score for side effects analysis."""
        if not side_effects_data:
            return 0.0

        confidence_metrics = side_effects_data.get("confidence_metrics", {})
        return round(confidence_metrics.get("overall_confidence", 0.0), 2)

    def _calculate_records_confidence(self, medical_records_data: Dict) -> float:
        """Calculate confidence score for medical records analysis."""
        if not medical_records_data:
            return 0.0

        data_quality = medical_records_data.get("data_quality", {})
        total_records = data_quality.get("total_records", 0)
        completeness = data_quality.get("completeness", 0.0)

        # Weight both factors
        quantity_score = min(total_records / 1000, 1.0)  # Normalize to max 1000 records
        quality_score = completeness

        return round((quantity_score * 0.4 + quality_score * 0.6), 2)

    def _aggregate_side_effects(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate side effects data from multiple sources."""
        try:
            aggregated_data = {
                "effects_summary": [],
                "severity_distribution": {},
                "frequency_analysis": {},
                "confidence_metrics": {
                    "overall_confidence": 0.0,
                    "data_completeness": 0.0,
                    "source_reliability": 0.0
                }
            }

            valid_results = [r for r in results if isinstance(r, dict) and not isinstance(r, Exception)]

            if not valid_results:
                return aggregated_data

            # Combine effects summaries
            all_effects = {}
            for result in valid_results:
                for effect in result.get("effects_summary", []):
                    effect_name = effect.get("effect")
                    if effect_name:
                        if effect_name not in all_effects:
                            all_effects[effect_name] = {
                                "effect": effect_name,
                                "count": 0,
                                "severity": [],
                                "sources": set()
                            }
                        all_effects[effect_name]["count"] += effect.get("count", 1)
                        all_effects[effect_name]["severity"].append(effect.get("severity", "unknown"))
                        all_effects[effect_name]["sources"].add(effect.get("source", "unknown"))

            # Convert aggregated effects to list format
            aggregated_data["effects_summary"] = [
                {
                    "effect": name,
                    "count": data["count"],
                    "severity": max(set(data["severity"]), key=data["severity"].count),
                    "sources": list(data["sources"])
                }
                for name, data in all_effects.items()
            ]

            # Sort by count descending
            aggregated_data["effects_summary"].sort(key=lambda x: x["count"], reverse=True)

            # Calculate confidence metrics
            total_effects = len(aggregated_data["effects_summary"])
            total_sources = len(set(source for effect in aggregated_data["effects_summary"] for source in effect["sources"]))

            aggregated_data["confidence_metrics"] = {
                "overall_confidence": round(min(total_effects / 100, 1.0) * 0.7 + min(total_sources / 3, 1.0) * 0.3, 2),
                "data_completeness": round(sum(1 for e in aggregated_data["effects_summary"] if len(e["sources"]) > 1) / max(total_effects, 1), 2),
                "source_reliability": round(total_sources / 3, 2)  # Normalized by expected number of sources
            }

            return aggregated_data

        except Exception as e:
            logger.error(f"Error aggregating side effects: {str(e)}")
            return {
                "effects_summary": [],
                "severity_distribution": {},
                "frequency_analysis": {},
                "confidence_metrics": {"error": str(e)}
            }

    def _merge_dictionaries(self, dict_list: List[Dict]) -> Dict:
        """Merge multiple dictionaries by summing their values."""
        merged = {}
        for d in dict_list:
            for k, v in d.items():
                if k not in merged:
                    merged[k] = 0
                merged[k] += v
        return merged
