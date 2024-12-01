import logging
from typing import Dict, List, Any
from datetime import datetime
from ..data_sources.fda_client import FDAClient
from ..config import REPORT_SETTINGS, FDA_API_KEY

logger = logging.getLogger(__name__)

class MedicalRecordsReportGenerator:
    """Generates medical records reports using FDA FAERS and public health data."""

    def __init__(self):
        self.fda_client = None
        self.max_records = REPORT_SETTINGS['max_medical_records']

    async def generate_medical_records_report(self, compound_name: str) -> Dict[str, Any]:
        """Generate medical records report using FDA FAERS data."""
        try:
            self.fda_client = FDAClient(api_key=FDA_API_KEY)

            # Get real adverse event records from FDA
            records = await self.fda_client.get_adverse_events(compound_name)

            if not records:
                logger.warning(f"No FDA records found for {compound_name}")
                return self._empty_report_template()

            return {
                "timestamp": datetime.now().isoformat(),
                "compound_name": compound_name,
                "data_source": "FDA FAERS Database",
                "summary_statistics": self._calculate_summary_statistics(records),
                "usage_patterns": self._analyze_usage_patterns(records),
                "outcome_analysis": self._analyze_outcomes(records),
                "demographic_distribution": self._analyze_demographics(records),
                "confidence_metrics": {
                    "total_records": len(records),
                    "data_completeness": self._calculate_completeness(records),
                    "temporal_coverage": self._calculate_temporal_coverage(records),
                    "data_source_reliability": 0.9  # FDA FAERS is considered highly reliable
                }
            }

        except Exception as e:
            logger.error(f"Error generating medical records report for {compound_name}: {str(e)}")
            return self._empty_report_template()

    def _calculate_summary_statistics(self, records: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics from FDA records."""
        total_reports = len(records)
        unique_patients = len(set(record.get('patient', {}).get('patientid') for record in records if record.get('patient')))

        return {
            "total_reports": total_reports,
            "unique_patients": unique_patients,
            "reporting_rate": round(total_reports / max(1, unique_patients), 2),
            "unique_reactions": len(set(record.get('reaction') for record in records if record.get('reaction')))
        }

    def _analyze_usage_patterns(self, records: List[Dict]) -> Dict[str, Any]:
        """Analyze usage patterns from FDA records."""
        drug_info = {}
        indications = {}

        for record in records:
            if 'drug' in record:
                dosage = record['drug'].get('dose', 'Unknown')
                drug_info[dosage] = drug_info.get(dosage, 0) + 1

                indication = record['drug'].get('indication', 'Unknown')
                indications[indication] = indications.get(indication, 0) + 1

        return {
            "common_dosages": sorted(drug_info.items(), key=lambda x: x[1], reverse=True)[:5],
            "common_indications": sorted(indications.items(), key=lambda x: x[1], reverse=True)[:5]
        }

    def _analyze_outcomes(self, records: List[Dict]) -> Dict[str, Any]:
        """Analyze patient outcomes from medical records."""
        outcomes = {
            "improved": 0,
            "unchanged": 0,
            "deteriorated": 0,
            "unknown": 0
        }

        complications = {}

        for record in records:
            outcome = record.get('outcome', 'unknown')
            outcomes[outcome] = outcomes.get(outcome, 0) + 1

            if record.get('complications'):
                for complication in record['complications']:
                    complications[complication] = complications.get(complication, 0) + 1

        return {
            "outcome_distribution": outcomes,
            "common_complications": sorted(complications.items(), key=lambda x: x[1], reverse=True)[:5]
        }

    def _analyze_demographics(self, records: List[Dict]) -> Dict[str, Any]:
        """Analyze demographic distribution from FDA records."""
        age_groups = {
            "0-18": 0,
            "19-30": 0,
            "31-50": 0,
            "51-70": 0,
            "70+": 0
        }

        gender_distribution = {}

        for record in records:
            patient = record.get('patient', {})
            age = patient.get('age')
            if age:
                if age <= 18:
                    age_groups["0-18"] += 1
                elif age <= 30:
                    age_groups["19-30"] += 1
                elif age <= 50:
                    age_groups["31-50"] += 1
                elif age <= 70:
                    age_groups["51-70"] += 1
                else:
                    age_groups["70+"] += 1

            gender = patient.get('gender')
            if gender:
                gender_distribution[gender] = gender_distribution.get(gender, 0) + 1

        return {
            "age_distribution": age_groups,
            "gender_distribution": gender_distribution
        }

    def _calculate_completeness(self, records: List[Dict]) -> float:
        """Calculate data completeness score for FDA records."""
        required_fields = ['patient', 'reaction', 'drug', 'outcome']
        scores = []

        for record in records:
            present_fields = sum(1 for field in required_fields if record.get(field))
            scores.append(present_fields / len(required_fields))

        return round(sum(scores) / len(scores), 2) if scores else 0

    def _calculate_temporal_coverage(self, records: List[Dict]) -> Dict[str, Any]:
        """Calculate temporal coverage of the medical records."""
        dates = [record.get('date') for record in records if record.get('date')]

        if not dates:
            return {
                "start_date": None,
                "end_date": None,
                "coverage_days": 0
            }

        start_date = min(dates)
        end_date = max(dates)
        coverage_days = (end_date - start_date).days

        return {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "coverage_days": coverage_days
        }

    def _empty_report_template(self) -> Dict[str, Any]:
        """Return empty template when no medical records found."""
        return {
            "timestamp": datetime.now().isoformat(),
            "compound_name": "",
            "summary_statistics": {
                "total_reports": 0,
                "unique_patients": 0,
                "reporting_rate": 0,
                "unique_reactions": 0
            },
            "usage_patterns": {
                "common_dosages": [],
                "common_indications": []
            },
            "outcome_analysis": {
                "outcome_distribution": {
                    "improved": 0,
                    "unchanged": 0,
                    "deteriorated": 0,
                    "unknown": 0
                },
                "common_complications": []
            },
            "demographic_distribution": {
                "age_distribution": {
                    "0-18": 0,
                    "19-30": 0,
                    "31-50": 0,
                    "51-70": 0,
                    "70+": 0
                },
                "gender_distribution": {}
            },
            "confidence_metrics": {
                "total_records": 0,
                "data_completeness": 0,
                "temporal_coverage": {
                    "start_date": None,
                    "end_date": None,
                    "coverage_days": 0
                }
            }
        }
