from typing import Dict, Any, List
import requests
import aiohttp
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
import asyncio

logger = logging.getLogger(__name__)

class SideEffectsAnalyzer:
    def __init__(self, fda_client=None):
        """Initialize the analyzer with multiple data source endpoints."""
        self.pubchem_base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        self.chembl_base_url = "https://www.ebi.ac.uk/chembl/api/data"
        self.sider_base_url = "http://sideeffects.embl.de/api/v1"
        self.faers_base_url = "https://api.fda.gov/drug/event.json"

        # Store FDA client if provided
        self.fda_client = fda_client
        self.use_faers = bool(fda_client)

        if not self.use_faers:
            logger.warning("FDA client not provided. FAERS data will not be available.")

    async def analyze_side_effects(self, similar_compounds: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze potential side effects based on similar compounds using multiple databases."""
        try:
            all_effects = []

            for compound in similar_compounds:
                compound_effects = []

                # Collect effects from all available sources
                if compound.get("source") == "PubChem" and compound.get("cid"):
                    effects = await self._get_pubchem_effects(compound["cid"])
                    compound_effects.extend(effects)

                if compound.get("source") == "ChEMBL" and compound.get("chembl_id"):
                    effects = await self._get_chembl_effects(compound["chembl_id"])
                    compound_effects.extend(effects)

                # Get SIDER effects using compound name
                if compound.get("iupac_name"):
                    effects = await self._get_sider_effects(compound["iupac_name"])
                    compound_effects.extend(effects)

                # Get FAERS effects using compound name
                if compound.get("iupac_name"):
                    effects = await self._get_faers_effects(compound["iupac_name"])
                    compound_effects.extend(effects)

                if compound_effects:
                    all_effects.append({
                        "compound_id": compound.get("cid") or compound.get("chembl_id"),
                        "compound_name": compound.get("iupac_name", "Unknown"),
                        "similarity": compound.get("similarity", 0.0),
                        "effects": compound_effects,
                        "total_effects": len(compound_effects),
                        "sources": list(set(effect["source"] for effect in compound_effects))
                    })

            # Generate comprehensive analysis
            effects_summary = self._generate_effects_summary(all_effects)
            severity_dist = self._analyze_severity_distribution(all_effects)
            frequency_dist = self._analyze_frequency_distribution(all_effects)

            # Calculate confidence metrics
            confidence_metrics = self._calculate_confidence_metrics(all_effects)

            return {
                "effects_summary": effects_summary,
                "detailed_effects": all_effects,
                "severity_distribution": severity_dist,
                "frequency_analysis": frequency_dist,
                "confidence_metrics": confidence_metrics,
                "data_sources": list(set(
                    source for compound in all_effects
                    for source in compound["sources"]
                )),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error analyzing side effects: {str(e)}")
            raise

    async def _get_pubchem_effects(self, compound_id: str) -> List[Dict[str, Any]]:
        """Fetch side effects data from PubChem."""
        try:
            response = requests.get(
                f"{self.pubchem_base_url}/compound/cid/{compound_id}/assaysummary/JSON"
            )
            response.raise_for_status()
            data = response.json()

            effects = []
            if "AssaySummaries" in data:
                for assay in data["AssaySummaries"]:
                    if "Outcome" in assay and "SideEffect" in assay.get("BioAssayType", ""):
                        effects.append({
                            "effect": assay.get("Name", "Unknown"),
                            "severity": self._classify_severity(assay.get("Score", 0)),
                            "frequency": self._extract_frequency(assay.get("Comment", "")),
                            "source": "PubChem",
                            "confidence": self._calculate_confidence(assay)
                        })
            return effects

        except Exception as e:
            logger.error(f"Error fetching PubChem effects: {str(e)}")
            return []

    async def _get_chembl_effects(self, compound_id: str) -> List[Dict[str, Any]]:
        """Fetch side effects data from ChEMBL."""
        try:
            response = requests.get(
                f"{self.chembl_base_url}/molecule/{compound_id}/adverse_effects"
            )
            response.raise_for_status()
            data = response.json()

            effects = []
            for effect in data.get("adverse_effects", []):
                effects.append({
                    "effect": effect.get("effect_name", "Unknown"),
                    "severity": effect.get("severity", "Unknown"),
                    "frequency": effect.get("frequency", "Unknown"),
                    "source": "ChEMBL",
                    "confidence": self._calculate_confidence(effect)
                })
            return effects

        except Exception as e:
            logger.error(f"Error fetching ChEMBL effects: {str(e)}")
            return []

    async def _get_sider_effects(self, compound_name: str) -> List[Dict[str, Any]]:
        """Fetch side effects data from SIDER database."""
        try:
            response = requests.get(
                f"{self.sider_base_url}/drug/{compound_name}/side-effects"
            )
            response.raise_for_status()
            data = response.json()

            effects = []
            for effect in data:
                effects.append({
                    "effect": effect.get("side_effect_name"),
                    "severity": self._classify_severity_from_sider(effect),
                    "frequency": effect.get("frequency", "Unknown"),
                    "source": "SIDER",
                    "confidence": self._calculate_confidence(effect)
                })
            return effects

        except Exception as e:
            logger.error(f"Error fetching SIDER effects: {str(e)}")
            return []

    async def _get_faers_effects(self, compound_name: str) -> List[Dict[str, Any]]:
        """Retrieve adverse effects data from FDA FAERS database."""
        if not self.use_faers or not self.fda_client:
            return []

        try:
            effects = []
            # Use FDA client to get both recent and historical data
            for timeframe in ["recent", "historical"]:
                params = {
                    'search': f'patient.drug.medicinalproduct:"{compound_name}"',
                    'count': 'patient.reaction.reactionmeddrapt.exact',
                    'limit': 100
                }

                if timeframe == "recent":
                    # Add date range for recent data (last 2 years)
                    current_year = datetime.now().year
                    params['search'] += f' AND receivedate:[{current_year-2}0101 TO {current_year}1231]'

                data = await self.fda_client.get_adverse_events(params)

                if not data or 'results' not in data:
                    continue

                for result in data['results']:
                    term = result.get('term')
                    count = result.get('count', 0)

                    if term:
                        # Get detailed outcome data
                        severity_data = await self._get_outcome_data(compound_name, term)

                        # Calculate confidence based on multiple factors
                        confidence_factors = {
                            'count': count,
                            'has_outcome': bool(severity_data),
                            'data_quality': self._assess_data_quality(result),
                            'is_recent': timeframe == "recent",
                            'has_demographic_data': bool(result.get('patient', {}).get('patientonsetage'))
                        }

                        effects.append({
                            'effect': term,
                            'count': count,
                            'frequency': self._calculate_frequency_from_count(count),
                            'severity': severity_data.get('severity', 'unknown'),
                            'outcome_details': severity_data.get('details', {}),
                            'source': 'FAERS',
                            'timeframe': timeframe,
                            'confidence': self._calculate_confidence(confidence_factors)
                        })

            return sorted(effects, key=lambda x: (x['count'], x['confidence']), reverse=True)

        except asyncio.TimeoutError:
            logger.error("FAERS API timeout")
            return []
        except Exception as e:
            logger.error(f"Error retrieving FAERS effects: {str(e)}")
            return []

    async def _get_outcome_data(self, compound_name: str, reaction_term: str) -> Dict[str, Any]:
        """Get detailed outcome data for a specific adverse reaction."""
        if not self.use_faers or not self.fda_client:
            return {}

        try:
            params = {
                'search': (
                    f'patient.drug.medicinalproduct:"{compound_name}" AND '
                    f'patient.reaction.reactionmeddrapt:"{reaction_term}"'
                ),
                'count': 'patient.reaction.reactionoutcome'
            }

            data = await self.fda_client.get_adverse_events(params)

            if not data or 'results' not in data:
                return {}

            outcomes = data.get('results', [])

            # Map outcome codes to severity levels
            severity_mapping = {
                '1': 'fatal',
                '2': 'life_threatening',
                '3': 'hospitalization',
                '4': 'disability',
                '5': 'congenital_anomaly',
                '6': 'other_serious',
                '7': 'not_serious'
            }

            outcome_counts = {
                severity_mapping.get(result.get('term', ''), 'unknown'): result.get('count', 0)
                for result in outcomes
            }

            total_reports = sum(outcome_counts.values())

            return {
                'severity': self._classify_severity_from_faers(outcome_counts),
                'details': {
                    'outcome_distribution': outcome_counts,
                    'total_reports': total_reports,
                    'serious_percentage': (
                        sum(count for severity, count in outcome_counts.items()
                            if severity != 'not_serious' and severity != 'unknown') / total_reports
                        if total_reports > 0 else 0
                    )
                }
            }

        except Exception as e:
            logger.error(f"Error retrieving outcome data: {str(e)}")
            return {}

    def _classify_severity_from_faers(self, result: Dict[str, Any]) -> str:
        """Enhanced severity classification of FAERS effects based on comprehensive outcome data."""
        outcomes = result.get('outcomes', [])
        term = result.get('term', '').lower()
        count = result.get('count', 0)

        # Critical terms that always indicate severe reactions
        critical_terms = {'death', 'fatal', 'life-threatening', 'anaphylaxis', 'cardiac arrest'}
        if any(crit in term for crit in critical_terms):
            return "Severe"

        # Classify based on outcomes
        if 'Death' in outcomes or 'Life-Threatening' in outcomes:
            return "Severe"
        elif 'Hospitalization' in outcomes or 'Disability' in outcomes:
            return "Moderate to Severe"
        elif 'Other Serious' in outcomes or count > 50:  # High frequency indicates higher severity
            return "Moderate"
        else:
            return "Mild to Moderate"

    def _assess_data_quality(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Assess the quality and completeness of effect data."""
        quality_score = 0.0
        completeness_score = 0.0

        # Check data completeness
        required_fields = ["term", "count", "serious_outcomes"]
        completeness_score = sum(1 for field in required_fields if data.get(field)) / len(required_fields)

        # Assess data quality based on source-specific criteria
        if data.get("source") == "FAERS":
            # Higher quality if more reports and better documentation
            report_count = data.get("count", 0)
            quality_score += min(0.5, report_count / 10000)  # Cap at 0.5 for report count
            quality_score += 0.3 if data.get("serious_outcomes") else 0.0
            quality_score += 0.2 if data.get("term_type") == "PT" else 0.0  # Preferred Term

        elif data.get("source") in ["PubChem", "ChEMBL", "SIDER"]:
            # Quality metrics for other sources
            quality_score += 0.4 if data.get("effect") else 0.0
            quality_score += 0.3 if data.get("severity") != "Unknown" else 0.0
            quality_score += 0.3 if data.get("frequency") != "Unknown" else 0.0

        final_quality = (quality_score + completeness_score) / 2
        return {
            "overall_quality": round(final_quality * 100, 1),
            "completeness": round(completeness_score * 100, 1),
            "reliability": round(quality_score * 100, 1)
        }

    def _calculate_confidence_metrics(self, all_effects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall confidence metrics for the analysis."""
        if not all_effects:
            return {
                "overall_confidence": 0.0,
                "data_coverage": 0.0,
                "source_reliability": 0.0
            }

        total_effects = sum(compound["total_effects"] for compound in all_effects)
        total_sources = len(set(
            source
            for compound in all_effects
            for source in compound["sources"]
        ))

        # Calculate average data quality across all effects
        quality_scores = []
        for compound in all_effects:
            for effect in compound["effects"]:
                if quality_data := self._assess_data_quality(effect):
                    quality_scores.append(quality_data["overall_quality"])

        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        return {
            "overall_confidence": round(avg_quality, 1),
            "data_coverage": round((total_effects / (len(all_effects) * 4)) * 100, 1),  # Normalize by expected 4 sources
            "source_reliability": round((total_sources / 4) * 100, 1),  # Normalize by total possible sources
            "total_effects_analyzed": total_effects,
            "unique_sources_used": total_sources
        }

    def _calculate_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate confidence score based on data quality and source."""
        base_score = 0.5
        modifiers = 0.0

        # Add source-specific confidence modifiers
        if data.get("source") == "PubChem":
            modifiers += 0.2 if data.get("Score") else 0.0
        elif data.get("source") == "ChEMBL":
            modifiers += 0.3 if data.get("assay_type") == "Approved_Drug" else 0.1
        elif data.get("source") == "SIDER":
            modifiers += 0.25 if data.get("medical_significance") else 0.1
        elif data.get("source") == "FAERS":
            modifiers += min(0.3, data.get("count", 0) / 10000)

        # Add data completeness modifier
        completeness = sum(1 for k in ["effect", "severity", "frequency"] if data.get(k)) / 3
        modifiers += completeness * 0.2

        final_score = min(1.0, base_score + modifiers)
        return round(final_score * 100, 1)  # Return as percentage

    def _calculate_frequency_from_count(self, count: int) -> str:
        """Calculate frequency category based on report count."""
        if count > 1000:
            return "Very Common"
        elif count > 100:
            return "Common"
        elif count > 10:
            return "Uncommon"
        elif count > 1:
            return "Rare"
        else:
            return "Very Rare"
