import asyncpg
import logging
from typing import Dict, List, Any
from ..config import MIMIC_DB_CONFIG

logger = logging.getLogger(__name__)

class MIMICClient:
    def __init__(self):
        self.config = MIMIC_DB_CONFIG
        self.pool = None

    async def initialize(self):
        """Initialize the database connection pool."""
        try:
            self.pool = await asyncpg.create_pool(**self.config)
            logger.info("Successfully connected to MIMIC database")
        except Exception as e:
            logger.error(f"Failed to connect to MIMIC database: {str(e)}")
            raise

    async def get_medical_records(self, compound_name: str) -> Dict[str, Any]:
        """Fetch anonymized medical records related to the compound."""
        if not self.pool:
            await self.initialize()

        try:
            async with self.pool.acquire() as conn:
                # Query prescriptions and outcomes
                records = await conn.fetch("""
                    SELECT
                        p.drug,
                        p.route,
                        p.dose_val_rx,
                        p.dose_unit_rx,
                        o.diagnosis,
                        o.severity,
                        d.age_at_admission,
                        d.gender
                    FROM prescriptions p
                    JOIN outcomes o ON p.hadm_id = o.hadm_id
                    JOIN demographic d ON p.subject_id = d.subject_id
                    WHERE LOWER(p.drug) LIKE LOWER($1)
                    LIMIT 1000
                """, f"%{compound_name}%")

                if not records:
                    return self._empty_records_template()

                return self._process_records(records)

        except Exception as e:
            logger.error(f"Error fetching medical records for {compound_name}: {str(e)}")
            return self._empty_records_template()

    def _process_records(self, records) -> Dict[str, Any]:
        """Process and analyze the medical records data."""
        usage_stats = self._calculate_usage_statistics(records)
        outcomes = self._analyze_outcomes(records)
        demographics = self._analyze_demographics(records)
        temporal = self._analyze_temporal_patterns(records)

        return {
            "usage_statistics": usage_stats,
            "outcome_analysis": outcomes,
            "demographic_distribution": demographics,
            "temporal_trends": temporal,
            "data_quality": {
                "total_records": len(records),
                "completeness": self._calculate_completeness(records),
                "time_range": self._calculate_time_range(records)
            }
        }

    def _empty_records_template(self) -> Dict[str, Any]:
        """Return empty template when no records found."""
        return {
            "usage_statistics": {},
            "outcome_analysis": {},
            "demographic_distribution": {},
            "temporal_trends": {},
            "data_quality": {
                "total_records": 0,
                "completeness": 0,
                "time_range": None
            }
        }

    def _calculate_usage_statistics(self, records) -> Dict[str, Any]:
        """Calculate usage statistics from records."""
        routes = {}
        doses = []

        for record in records:
            if record['route']:
                routes[record['route']] = routes.get(record['route'], 0) + 1
            if record['dose_val_rx']:
                doses.append(float(record['dose_val_rx']))

        return {
            "administration_routes": routes,
            "average_dose": sum(doses) / len(doses) if doses else None,
            "dose_range": {
                "min": min(doses) if doses else None,
                "max": max(doses) if doses else None
            }
        }

    def _analyze_outcomes(self, records) -> Dict[str, Any]:
        """Analyze patient outcomes."""
        diagnoses = {}
        severities = {}

        for record in records:
            if record['diagnosis']:
                diagnoses[record['diagnosis']] = diagnoses.get(record['diagnosis'], 0) + 1
            if record['severity']:
                severities[record['severity']] = severities.get(record['severity'], 0) + 1

        return {
            "diagnoses_distribution": diagnoses,
            "severity_distribution": severities
        }

    def _analyze_demographics(self, records) -> Dict[str, Any]:
        """Analyze demographic distribution."""
        ages = []
        genders = {}

        for record in records:
            if record['age_at_admission']:
                ages.append(record['age_at_admission'])
            if record['gender']:
                genders[record['gender']] = genders.get(record['gender'], 0) + 1

        return {
            "age_distribution": {
                "average": sum(ages) / len(ages) if ages else None,
                "range": {
                    "min": min(ages) if ages else None,
                    "max": max(ages) if ages else None
                }
            },
            "gender_distribution": genders
        }

    def _analyze_temporal_patterns(self, records) -> Dict[str, Any]:
        """Analyze temporal patterns in usage and outcomes."""
        try:
            # Group records by diagnosis and analyze patterns
            diagnosis_trends = {}
            severity_trends = {}
            usage_trends = {}

            for record in records:
                diagnosis = record['diagnosis']
                severity = record['severity']

                if diagnosis:
                    if diagnosis not in diagnosis_trends:
                        diagnosis_trends[diagnosis] = {
                            'count': 0,
                            'severity_distribution': {},
                            'routes': {}
                        }

                    diagnosis_trends[diagnosis]['count'] += 1

                    if severity:
                        if severity not in diagnosis_trends[diagnosis]['severity_distribution']:
                            diagnosis_trends[diagnosis]['severity_distribution'][severity] = 0
                        diagnosis_trends[diagnosis]['severity_distribution'][severity] += 1

                    if record['route']:
                        if record['route'] not in diagnosis_trends[diagnosis]['routes']:
                            diagnosis_trends[diagnosis]['routes'][record['route']] = 0
                        diagnosis_trends[diagnosis]['routes'][record['route']] += 1

            return {
                "diagnosis_patterns": diagnosis_trends,
                "record_distribution": {
                    "total_records": len(records),
                    "unique_diagnoses": len(diagnosis_trends),
                    "data_completeness": self._calculate_completeness(records)
                },
                "temporal_metrics": {
                    "has_temporal_data": bool(records),
                    "patterns_identified": bool(diagnosis_trends)
                }
            }

        except Exception as e:
            logger.error(f"Error analyzing temporal patterns: {str(e)}")
            return {
                "diagnosis_patterns": {},
                "record_distribution": {
                    "total_records": 0,
                    "unique_diagnoses": 0,
                    "data_completeness": 0
                },
                "temporal_metrics": {
                    "has_temporal_data": False,
                    "patterns_identified": False
                }
            }

    def _calculate_completeness(self, records) -> float:
        """Calculate data completeness score."""
        required_fields = ['drug', 'route', 'dose_val_rx', 'diagnosis']
        completeness_scores = []

        for record in records:
            fields_present = sum(1 for field in required_fields if record[field])
            completeness_scores.append(fields_present / len(required_fields))

        return sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0

    def _calculate_time_range(self, records) -> Dict[str, Any]:
        """Calculate the time range of the records."""
        # This would typically calculate the actual time range
        # For now, return a simplified version
        return {
            "has_time_data": bool(records),
            "record_count": len(records)
        }
