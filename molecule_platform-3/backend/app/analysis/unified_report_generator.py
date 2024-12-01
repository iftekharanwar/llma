from typing import Dict, List, Any, Optional
import asyncio
import logging
from datetime import datetime
from ..data_sources.fda_client import FDAClient
from ..data_sources.pubchem_client import PubChemClient
from ..data_sources.chembl_client import ChEMBLClient
from .comprehensive_report_generator import ComprehensiveReportGenerator
from .medical_records_report import MedicalRecordsReportGenerator
from .side_effects_report import SideEffectsReportGenerator
from .rdkit_risk_assessment import RDKitRiskAssessment
from ..config import REPORT_SETTINGS, FDA_API_KEY

logger = logging.getLogger(__name__)

class UnifiedReportGenerator:
    """Generates unified comprehensive reports combining all analysis components."""

    def __init__(self):
        self.comprehensive_generator = ComprehensiveReportGenerator()
        self.medical_records_generator = MedicalRecordsReportGenerator()
        self.side_effects_generator = SideEffectsReportGenerator()
        self.rdkit_risk_assessor = RDKitRiskAssessment()
        self.min_confidence = REPORT_SETTINGS['min_confidence_score']
        self.max_retries = 3
        self.retry_delay = 2  # seconds

        # Initialize clients with proper API keys
        self.fda_client = None
        self.pubchem_client = PubChemClient()
        self.chembl_client = ChEMBLClient()

    async def generate_unified_report(
        self,
        compound_name: str,
        smiles: str,
        context: str = "pharmaceutical"
    ) -> Dict[str, Any]:
        """Generate a comprehensive unified report for a given compound."""
        try:
            # Initialize FDA client with API key from environment
            self.fda_client = FDAClient(api_key=FDA_API_KEY)

            # Initialize parallel tasks for data retrieval with proper error handling
            tasks = [
                self._fetch_with_retry(
                    self.comprehensive_generator.generate_report,
                    smiles,
                    "Comprehensive analysis"
                ),
                self._fetch_with_retry(
                    self.rdkit_risk_assessor.assess_molecule,
                    smiles,
                    "RDKit analysis"
                ),
                self._fetch_with_retry(
                    self.side_effects_generator.generate_side_effects_report,
                    smiles=smiles,
                    compound_name=compound_name,
                    api_key=FDA_API_KEY,
                    description="Side effects analysis"
                ),
                self._fetch_with_retry(
                    self.medical_records_generator.generate_medical_records_report,
                    compound_name,
                    "Medical records analysis"
                )
            ]

            # Wait for all tasks with timeout
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
            except asyncio.TimeoutError:
                logger.error("Timeout while gathering report data")
                return self._empty_report_template(compound_name, smiles, context)

            # Process results with enhanced error handling
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error in report component {i}: {str(result)}")
                    processed_results.append({"error": str(result)})
                else:
                    processed_results.append(result)

            # Generate section-specific reports with confidence scoring
            similarity_report = self._process_similarity_data(processed_results[0])
            risk_report = self._process_risk_data(processed_results[1])
            side_effects_report = self._process_side_effects_data(processed_results[2])
            medical_report = self._process_medical_data(processed_results[3])

            # Generate context-specific analysis
            context_analysis = self._generate_context_analysis(
                context,
                processed_results[1],
                processed_results[2],
                processed_results[3]
            )

            # Calculate unified confidence metrics
            confidence_metrics = self._calculate_unified_confidence([
                similarity_report,
                risk_report,
                side_effects_report,
                medical_report
            ])

            # Track active data sources
            active_sources = self._get_active_data_sources({
                "similarity": processed_results[0],
                "risk": processed_results[1],
                "side_effects": processed_results[2]
            })

            return {
                "timestamp": datetime.now().isoformat(),
                "compound_info": {
                    "name": compound_name,
                    "smiles": smiles,
                    "context": context
                },
                "similarity_analysis": similarity_report,
                "risk_assessment": risk_report,
                "side_effects_analysis": side_effects_report,
                "medical_records_analysis": medical_report,
                "context_specific_analysis": context_analysis,
                "confidence_metrics": confidence_metrics,
                "data_sources": active_sources,
                "report_metadata": {
                    "version": "2.0",
                    "generated_at": datetime.now().isoformat(),
                    "data_freshness": datetime.now().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"Error generating unified report: {str(e)}")
            return self._empty_report_template(compound_name, smiles, context)

    def _process_similarity_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and format similarity analysis data."""
        if not data or "error" in data:
            return {"error": "Similarity analysis failed"}

        similar_compounds = data.get("similar_compounds", [])

        # Process and enrich similar compounds data
        processed_compounds = []
        for compound in similar_compounds[:5]:  # Limit to top 5 most similar
            processed = {
                "name": compound.get("name", "Unknown"),
                "smiles": compound.get("smiles"),
                "similarity_score": compound.get("similarity", 0),
                "source": compound.get("source"),
                "properties": compound.get("properties", {}),
                "bioactivity": compound.get("bioactivity", {}),
                "confidence": {
                    "data_completeness": compound.get("confidence_metrics", {}).get("data_completeness", 0),
                    "property_reliability": compound.get("confidence_metrics", {}).get("property_reliability", 0),
                    "has_bioactivity": compound.get("confidence_metrics", {}).get("has_bioactivity", 0)
                }
            }
            processed_compounds.append(processed)

        return {
            "similar_compounds": processed_compounds,
            "analysis_summary": {
                "total_compounds_found": len(similar_compounds),
                "average_similarity": sum(c.get("similarity", 0) for c in similar_compounds) / len(similar_compounds) if similar_compounds else 0,
                "data_sources": [c.get("source") for c in similar_compounds if c.get("source")],  # Changed from set() to list comprehension
                "property_coverage": self._calculate_property_coverage(processed_compounds)
            },
            "confidence_score": self._calculate_section_confidence(data)
        }

    def _process_side_effects_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and format side effects analysis data."""
        if not data or "error" in data:
            return {"error": "Side effects analysis failed"}

        # Extract FDA FAERS data
        faers_data = data.get("faers_data", {})
        events_summary = faers_data.get("events_summary", {})
        severity_dist = faers_data.get("severity_distribution", {})
        demographic_data = faers_data.get("demographic_data", {})

        # Process drug label warnings
        label_data = data.get("drug_label", {})
        warnings = label_data.get("warnings", [])
        adverse_reactions = label_data.get("adverse_reactions", [])

        # Combine and process all side effects data
        processed_effects = []
        for effect in events_summary.get("most_common", []):
            effect_name, count = effect
            processed_effects.append({
                "effect": effect_name,
                "frequency": events_summary.get("event_frequency", {}).get(effect_name, 0),
                "severity": self._determine_severity(effect_name, severity_dist),
                "demographic_impact": self._extract_demographic_impact(effect_name, demographic_data),
                "confidence": self._calculate_effect_confidence({
                    "report_count": count,
                    "has_severity_data": bool(severity_dist),
                    "has_demographic_data": bool(demographic_data),
                    "in_label_warnings": effect_name.lower() in [w.lower() for w in warnings + adverse_reactions]
                })
            })

        return {
            "effects_summary": {
                "total_reports": events_summary.get("total_events", 0),
                "unique_effects": events_summary.get("unique_events", 0),
                "severity_distribution": severity_dist,
                "demographic_analysis": demographic_data
            },
            "detailed_effects": processed_effects,
            "label_warnings": {
                "warnings": warnings,
                "adverse_reactions": adverse_reactions
            },
            "confidence_metrics": {
                "data_completeness": faers_data.get("confidence_metrics", {}).get("data_completeness", 0),
                "reporting_quality": faers_data.get("confidence_metrics", {}).get("reporting_quality", 0),
                "sample_size_confidence": faers_data.get("confidence_metrics", {}).get("sample_size_confidence", 0)
            }
        }

    def _process_risk_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and format risk assessment data."""
        if not data or "error" in data:
            return {"error": "Risk assessment failed"}

        # Extract and process risk factors with confidence scores
        risk_factors = []
        for risk in data.get("toxicity_risks", {}).get("identified_risks", []):
            processed_risk = {
                "type": risk.get("type"),
                "severity": risk.get("severity", 0),
                "description": risk.get("description"),
                "confidence": risk.get("confidence", 0),
                "supporting_data": risk.get("supporting_data", [])
            }
            risk_factors.append(processed_risk)

        # Calculate weighted risk scores
        risk_scores = data.get("risk_scores", {})
        weighted_score = sum(
            score * weight
            for score, weight in zip(
                risk_scores.get("component_scores", {}).values(),
                risk_scores.get("weights", {}).values()
            )
        ) if risk_scores.get("component_scores") and risk_scores.get("weights") else 0

        return {
            "risk_factors": risk_factors,
            "risk_scores": {
                "overall_risk": risk_scores.get("overall_risk", 0),
                "weighted_score": weighted_score,
                "component_scores": risk_scores.get("component_scores", {}),
                "confidence": risk_scores.get("confidence", 0)
            },
            "property_analysis": {
                "molecular_properties": data.get("molecular_properties", {}),
                "structural_alerts": data.get("structural_alerts", []),
                "prediction_reliability": data.get("prediction_reliability", 0)
            },
            "recommendations": self._process_risk_recommendations(data.get("recommendations", [])),
            "confidence_score": self._calculate_section_confidence(data)
        }

    def _process_side_effects_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and format side effects data."""
        if not data or "error" in data:
            return {"error": "Side effects analysis failed"}

        return {
            "common_effects": data.get("effects_summary", {}).get("most_common_effects", [])[:5],
            "severity_distribution": data.get("severity_analysis", {}).get("distribution", {}),
            "confidence_score": data.get("confidence_metrics", {}).get("confidence_score", 0)
        }

    def _process_medical_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and format medical records data."""
        if not data or "error" in data:
            return {"error": "Medical records analysis failed"}

        return {
            "usage_summary": data.get("summary_statistics", {}),
            "outcomes": data.get("outcome_analysis", {}),
            "demographics": data.get("demographic_distribution", {}),
            "confidence_score": data.get("confidence_metrics", {}).get("data_completeness", 0)
        }

    def _generate_context_analysis(
        self,
        context: str,
        risk_data: Dict[str, Any],
        side_effects_data: Dict[str, Any],
        medical_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate context-specific analysis based on the intended use."""
        context_weights = REPORT_SETTINGS['context_weights'].get(context, {})
        if not context_weights:
            return {"error": f"Unsupported context: {context}"}

        scores = {}
        for metric, weight in context_weights.items():
            scores[metric] = self._calculate_context_metric(
                metric,
                risk_data,
                side_effects_data,
                medical_data
            )

        return {
            "context": context,
            "scores": scores,
            "overall_score": sum(score * weight for score, weight in zip(scores.values(), context_weights.values())),
            "recommendations": self._generate_context_recommendations(context, scores)
        }

    def _calculate_context_metric(
        self,
        metric: str,
        risk_data: Dict[str, Any],
        side_effects_data: Dict[str, Any],
        medical_data: Dict[str, Any]
    ) -> float:
        """Calculate specific metric score based on context."""
        if metric == "toxicity":
            return 1 - (risk_data.get("risk_scores", {}).get("overall_risk", 0) / 100)
        elif metric == "efficacy":
            outcomes = medical_data.get("outcome_analysis", {}).get("effectiveness", {})
            return outcomes.get("improved", 0) / sum(outcomes.values()) if sum(outcomes.values()) > 0 else 0
        elif metric == "side_effects":
            severity = side_effects_data.get("severity_analysis", {}).get("distribution", {})
            total = sum(severity.values())
            return 1 - (severity.get("severe", 0) / total if total > 0 else 0)
        elif metric == "environmental_impact":
            return 1 - (risk_data.get("environmental_risks", {}).get("overall_impact", 0) / 100)
        elif metric == "reactivity":
            return 1 - (risk_data.get("reactivity_scores", {}).get("overall_reactivity", 0) / 100)
        elif metric == "stability":
            return medical_data.get("stability_analysis", {}).get("stability_score", 0.5)

        return 0.5  # Default score for unknown metrics

    def _generate_context_recommendations(self, context: str, scores: Dict[str, float]) -> List[str]:
        """Generate context-specific recommendations based on scores."""
        recommendations = []

        for metric, score in scores.items():
            if score < 0.6:
                recommendations.append(
                    f"Consider improving {metric} for {context} applications (current score: {score:.2f})"
                )

        return recommendations

    def _calculate_unified_confidence(self, reports: List[Dict]) -> Dict:
        """Calculate unified confidence metrics across all report components."""
        try:
            confidence_scores = {
                'similarity_confidence': reports[0].get('confidence_metrics', {}).get('data_completeness', 0),
                'risk_confidence': reports[1].get('confidence_metrics', {}).get('assessment_confidence', 0),
                'side_effects_confidence': reports[2].get('confidence_metrics', {}).get('data_completeness', 0),
                'medical_records_confidence': reports[3].get('confidence_metrics', {}).get('data_completeness', 0)
            }

            # Weight the confidence scores based on data quality and completeness
            weighted_confidence = (
                confidence_scores['similarity_confidence'] * 0.3 +
                confidence_scores['risk_confidence'] * 0.3 +
                confidence_scores['side_effects_confidence'] * 0.3 +
                confidence_scores['medical_records_confidence'] * 0.1
            )

            return {
                'component_confidence': confidence_scores,
                'overall_confidence': round(weighted_confidence * 100, 2),
                'data_quality_metrics': {
                    'completeness': self._calculate_property_coverage(reports),
                    'reliability': self._determine_data_reliability(reports),
                    'source_diversity': len([r for r in reports if r.get('confidence_metrics', {}).get('data_completeness', 0) > 0.5])
                }
            }
        except Exception as e:
            logger.error(f"Error calculating unified confidence: {str(e)}")
            return {
                'component_confidence': {},
                'overall_confidence': 0,
                'data_quality_metrics': {'completeness': 0, 'reliability': 0, 'source_diversity': 0}
            }

    def _calculate_section_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate confidence score for a report section."""
        if not data or "error" in data:
            return 0.0

        metrics = []

        # Check data source reliability with updated weights
        if "data_source" in data:
            source_reliability = {
                "PubChem": 0.95,    # High reliability for structural and property data
                "ChEMBL": 0.95,     # High reliability for bioactivity data
                "FDA FAERS": 0.90,  # High reliability for adverse events
                "RDKit": 0.85,      # Good reliability for computed properties
                "Medical Records": 0.80  # Moderate reliability due to data limitations
            }
            metrics.append(source_reliability.get(data["data_source"], 0.5))

        # Check data completeness with enhanced metrics
        if "confidence_metrics" in data:
            confidence_data = data["confidence_metrics"]
            metrics.extend([
                confidence_data.get("data_completeness", 0),
                confidence_data.get("property_reliability", 0),
                confidence_data.get("has_bioactivity", 0),
                confidence_data.get("reporting_quality", 0)
            ])

        # Add bioactivity confidence if available
        if "bioactivity" in data:
            bioactivity = data["bioactivity"]
            if bioactivity and isinstance(bioactivity, dict):
                total_assays = bioactivity.get("total_assays", 0)
                if total_assays > 0:
                    metrics.append(min(1.0, total_assays / 100))  # Normalize to max 1.0

        return sum(metrics) / len(metrics) if metrics else 0.0

    def _determine_severity(self, effect: str, severity_dist: Dict[str, int]) -> str:
        """Determine severity level of a side effect."""
        total = sum(severity_dist.values())
        if total == 0:
            return "unknown"

        # Calculate weighted severity score
        severity_weights = {
            "mild": 1,
            "moderate": 2,
            "severe": 3
        }

        weighted_score = sum(count * severity_weights[level] for level, count in severity_dist.items()) / total

        if weighted_score <= 1.5:
            return "mild"
        elif weighted_score <= 2.5:
            return "moderate"
        else:
            return "severe"

    def _extract_demographic_impact(self, effect: str, demographic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract demographic impact for a specific side effect."""
        return {
            "age_groups": demographic_data.get("age_distribution", {}),
            "gender_distribution": demographic_data.get("gender_distribution", {})
        }

    def _calculate_effect_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate confidence score for a specific side effect."""
        confidence_factors = []

        # Report count factor (normalize to max of 100 reports)
        report_count = min(data.get("report_count", 0) / 100, 1.0)
        confidence_factors.append(report_count)

        # Data completeness factors
        if data.get("has_severity_data"):
            confidence_factors.append(0.3)
        if data.get("has_demographic_data"):
            confidence_factors.append(0.3)
        if data.get("in_label_warnings"):
            confidence_factors.append(0.4)

        return round(sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0, 2)

    def _process_risk_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and prioritize risk-based recommendations."""
        if not recommendations:
            return []

        processed_recommendations = []
        for rec in recommendations:
            if isinstance(rec, dict):
                processed_rec = {
                    "priority": rec.get("priority", "medium"),
                    "category": rec.get("category", "general"),
                    "recommendation": rec.get("text", rec.get("recommendation", "")),
                    "rationale": rec.get("rationale", ""),
                    "confidence": rec.get("confidence", 0.5)
                }
                processed_recommendations.append(processed_rec)
            elif isinstance(rec, str):
                processed_recommendations.append({
                    "priority": "medium",
                    "category": "general",
                    "recommendation": rec,
                    "rationale": "",
                    "confidence": 0.5
                })

        # Sort by priority
        priority_weights = {"high": 3, "medium": 2, "low": 1}
        return sorted(
            processed_recommendations,
            key=lambda x: (priority_weights.get(x["priority"], 0), x["confidence"]),
            reverse=True
        )

    async def _fetch_with_retry(self, func, *args, description="", **kwargs) -> Any:
        """Helper method to retry API calls with exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed {description} after {self.max_retries} attempts: {str(e)}")
                    raise
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(f"{description} failed, attempt {attempt + 1}/{self.max_retries}. Retrying in {wait_time}s")
                await asyncio.sleep(wait_time)

    def _get_active_data_sources(self, results: Dict[str, Any]) -> List[str]:
        """Track which real data sources are being used in the report."""
        sources = set()

        # Add data sources based on actual API responses
        if results.get("similarity"):
            if self.pubchem_client.last_response:
                sources.add("PubChem")
            if self.chembl_client.last_response:
                sources.add("ChEMBL")

        if results.get("side_effects"):
            if self.fda_client.last_response:
                sources.add("FDA FAERS")

        if results.get("risk"):
            sources.add("RDKit")  # Always used for molecular property calculations

        return sorted(list(sources))

    def _calculate_property_coverage(self, compounds: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate the coverage of different properties across similar compounds."""
        if not compounds:
            return {}

        # Define key properties to track
        properties = {
            "molecular_weight": 0,
            "alogp": 0,
            "psa": 0,
            "complexity": 0,
            "hbond_donors": 0,
            "hbond_acceptors": 0,
            "rotatable_bonds": 0
        }

        # Count availability of each property
        for compound in compounds:
            comp_props = compound.get("properties", {})
            for prop in properties:
                if comp_props.get(prop) is not None:
                    properties[prop] += 1

        # Convert counts to percentages
        total = len(compounds)
        return {prop: count / total for prop, count in properties.items()}

    def _determine_data_reliability(self, reports: List[Dict]) -> float:
        """Calculate reliability score based on data source quality and completeness."""
        try:
            reliability_scores = []

            # Check data source reliability
            for report in reports:
                if not report or not isinstance(report, dict):
                    continue

                metrics = report.get('confidence_metrics', {})
                if not metrics:
                    continue

                # Weight factors for reliability
                completeness = metrics.get('data_completeness', 0)
                has_validation = 1.0 if metrics.get('validation_performed', False) else 0.0
                source_quality = metrics.get('source_reliability', 0.8)  # Default to 0.8 for established sources

                # Calculate weighted reliability score
                reliability = (completeness * 0.4 + has_validation * 0.3 + source_quality * 0.3)
                reliability_scores.append(reliability)

            return round(sum(reliability_scores) / len(reliability_scores), 2) if reliability_scores else 0.0

        except Exception as e:
            logger.error(f"Error calculating data reliability: {str(e)}")
            return 0.0

    def _empty_report_template(self, compound_name: str, smiles: str, context: str) -> Dict[str, Any]:
        """Return empty template when report generation fails."""
        return {
            "timestamp": datetime.now().isoformat(),
            "compound_info": {
                "name": compound_name,
                "smiles": smiles,
                "context": context
            },
            "similarity_analysis": {"error": "Analysis failed"},
            "risk_assessment": {"error": "Assessment failed"},
            "side_effects_analysis": {"error": "Analysis failed"},
            "medical_records_analysis": {"error": "Analysis failed"},
            "context_specific_analysis": {"error": "Analysis failed"},
            "confidence_metrics": {
                "section_scores": {
                    "similarity_analysis": 0,
                    "risk_assessment": 0,
                    "side_effects": 0,
                    "medical_records": 0
                },
                "overall_confidence": 0
            }
        }
