import os
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime
import logging
import pubchempy as pcp
from chembl_webresource_client.new_client import new_client

logger = logging.getLogger(__name__)

class MedicalRecordsAnalyzer:
    def __init__(self):
        # Initialize connection to MIMIC database if available
        self.mimic_db_url = os.getenv('MIMIC_DB_URL')
        self.use_mimic = bool(self.mimic_db_url)

        if self.use_mimic:
            self.engine = create_engine(self.mimic_db_url)

        # Initialize alternative data sources
        self.molecule_client = new_client.molecule
        self.drug_client = new_client.drug_indication

    async def analyze_medical_records(self, similar_compounds: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze medical records for similar compounds using available data sources."""
        try:
            if self.use_mimic:
                return await self._analyze_mimic_records(similar_compounds)
            else:
                return await self._analyze_alternative_sources(similar_compounds)

        except Exception as e:
            logger.error(f"Error analyzing medical records: {str(e)}")
            raise

    async def _analyze_alternative_sources(self, similar_compounds: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze medical usage patterns using PubChem and ChEMBL data."""
        try:
            records = []
            for compound in similar_compounds:
                # Get PubChem data
                pcp_results = pcp.get_compounds(compound["name"], 'name')
                if pcp_results:
                    compound_data = pcp_results[0]
                    medical_uses = compound_data.iupac_name if hasattr(compound_data, 'iupac_name') else ''

                # Get ChEMBL data
                chembl_results = self.molecule_client.filter(pref_name__iexact=compound["name"])
                if chembl_results:
                    molecule = chembl_results[0]
                    indications = self.drug_client.filter(molecule_chembl_id=molecule['molecule_chembl_id'])

                    for indication in indications:
                        records.append({
                            'name': compound["name"],
                            'indication': indication['max_phase_for_ind'],
                            'mesh_heading': indication.get('mesh_heading', ''),
                            'efo_term': indication.get('efo_term', ''),
                            'source': 'ChEMBL'
                        })

            # Convert to DataFrame for analysis
            df = pd.DataFrame(records)

            return {
                "summary": self._generate_alternative_summary(df),
                "usage_statistics": self._analyze_alternative_usage(df),
                "outcome_analysis": self._analyze_alternative_outcomes(df),
                "temporal_trends": {"yearly_usage": {}, "usage_trend": "Data from chemical databases"},
                "demographic_distribution": {"note": "Demographic data not available from chemical databases"}
            }

        except Exception as e:
            logger.error(f"Error analyzing alternative sources: {str(e)}")
            return self._generate_empty_analysis()

    def _analyze_alternative_usage(self, records: pd.DataFrame) -> Dict[str, Any]:
        """Analyze usage patterns from chemical databases."""
        if records.empty:
            return {
                "total_compounds": 0,
                "unique_indications": 0,
                "indication_distribution": {}
            }

        return {
            "total_compounds": len(records),
            "unique_indications": records['indication'].nunique(),
            "indication_distribution": records['indication'].value_counts().to_dict()
        }

    def _analyze_alternative_outcomes(self, records: pd.DataFrame) -> Dict[str, Any]:
        """Analyze outcomes from chemical databases."""
        if records.empty:
            return {
                "max_phase_distribution": {},
                "therapeutic_areas": {},
                "development_status": "No data available"
            }

        return {
            "max_phase_distribution": records['indication'].value_counts().to_dict(),
            "therapeutic_areas": records['mesh_heading'].value_counts().head(5).to_dict(),
            "development_status": self._determine_development_status(records)
        }

    def _determine_development_status(self, records: pd.DataFrame) -> str:
        if records.empty:
            return "No data available"
        max_phase = records['indication'].max()
        if max_phase >= 4:
            return "Approved"
        elif max_phase >= 3:
            return "Clinical trials phase 3"
        elif max_phase >= 2:
            return "Clinical trials phase 2"
        elif max_phase >= 1:
            return "Clinical trials phase 1"
        return "Preclinical"

    def _generate_alternative_summary(self, records: pd.DataFrame) -> str:
        if records.empty:
            return "No medical usage data available from chemical databases"

        return f"""
        Based on chemical database analysis:
        - Total compounds analyzed: {len(records)}
        - Unique therapeutic indications: {records['indication'].nunique()}
        - Development status: {self._determine_development_status(records)}
        - Primary therapeutic areas: {', '.join(records['mesh_heading'].value_counts().head(3).index.tolist())}
        """

    async def _analyze_mimic_records(self, similar_compounds: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze medical records from MIMIC database for similar compounds."""
        try:
            # Create compound name list for SQL query
            compound_names = [comp["name"].lower() for comp in similar_compounds]
            compound_list = "','".join(compound_names)

            # Query MIMIC database for prescriptions and outcomes
            query = text(f"""
                WITH prescription_data AS (
                    SELECT
                        p.drug,
                        p.startdate,
                        p.enddate,
                        adm.admission_type,
                        adm.discharge_location,
                        pat.gender,
                        EXTRACT(YEAR FROM adm.admittime) - pat.anchor_year + pat.anchor_age as age,
                        p.hadm_id,
                        row_number() OVER (PARTITION BY p.hadm_id ORDER BY p.startdate) as rx_seq
                    FROM prescriptions p
                    JOIN admissions adm ON p.hadm_id = adm.hadm_id
                    JOIN patients pat ON adm.subject_id = pat.subject_id
                    WHERE LOWER(p.drug) IN ('{compound_list}')
                )
                SELECT
                    drug,
                    admission_type,
                    discharge_location,
                    gender,
                    age,
                    COUNT(DISTINCT hadm_id) as total_prescriptions,
                    AVG(CASE WHEN rx_seq > 1 THEN 1 ELSE 0 END) as readmission_rate,
                    COUNT(DISTINCT EXTRACT(YEAR FROM startdate)) as years_in_use
                FROM prescription_data
                GROUP BY drug, admission_type, discharge_location, gender, age
            """)

            with self.engine.connect() as conn:
                result = pd.read_sql(query, conn)

            if result.empty:
                logger.warning("No MIMIC records found for the compounds")
                return self._generate_empty_analysis()

            # Generate comprehensive analysis
            return {
                "usage_statistics": self._analyze_usage(result),
                "outcome_analysis": self._analyze_outcomes(result),
                "temporal_trends": self._analyze_trends(result),
                "demographic_distribution": self._analyze_demographics(result),
                "data_sources": ["MIMIC-III", "MIMIC-IV"],
                "confidence_metrics": {
                    "overall_confidence": self._calculate_confidence(result),
                    "data_completeness": len(result) / len(similar_compounds),
                    "temporal_coverage": result['years_in_use'].max()
                }
            }

        except Exception as e:
            logger.error(f"Error analyzing MIMIC records: {str(e)}")
            return self._generate_empty_analysis()

    def _analyze_usage(self, records: pd.DataFrame) -> Dict[str, Any]:
        """Analyze usage patterns from MIMIC records."""
        if records.empty:
            return self._generate_empty_analysis()["usage_statistics"]

        return {
            "total_prescriptions": int(records['total_prescriptions'].sum()),
            "admission_types": records.groupby('admission_type')['total_prescriptions'].sum().to_dict(),
            "average_duration": self._calculate_avg_duration(records),
            "readmission_rate": float(records['readmission_rate'].mean()),
            "years_in_use": int(records['years_in_use'].max())
        }

    def _analyze_outcomes(self, records: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patient outcomes from MIMIC records."""
        if records.empty:
            return self._generate_empty_analysis()["outcome_analysis"]

        discharge_outcomes = records.groupby('discharge_location')['total_prescriptions'].sum()

        return {
            "discharge_distribution": discharge_outcomes.to_dict(),
            "readmission_analysis": {
                "overall_rate": float(records['readmission_rate'].mean()),
                "by_admission_type": records.groupby('admission_type')['readmission_rate'].mean().to_dict()
            },
            "treatment_duration": {
                "average_days": self._calculate_avg_duration(records),
                "by_admission_type": records.groupby('admission_type').apply(self._calculate_avg_duration).to_dict()
            },
            "success_rate": float((discharge_outcomes['HOME'] / discharge_outcomes.sum()) if 'HOME' in discharge_outcomes else 0)
        }

    def _analyze_trends(self, records: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal trends from MIMIC records."""
        if records.empty:
            return self._generate_empty_analysis()["temporal_trends"]

        yearly_usage = self._calculate_trend(records)

        # Calculate trend direction
        trend_direction = self._determine_trend_direction(yearly_usage)

        # Calculate seasonal patterns
        seasonal_patterns = self._analyze_seasonal_patterns(records) if not records.empty else {}

        return {
            "yearly_usage": yearly_usage,
            "usage_trend": trend_direction,
            "seasonal_patterns": seasonal_patterns,
            "long_term_analysis": {
                "total_years": int(records['years_in_use'].max()),
                "usage_stability": "stable" if records['total_prescriptions'].std() / records['total_prescriptions'].mean() < 0.5 else "variable"
            }
        }

    def _analyze_demographics(self, records: pd.DataFrame) -> Dict[str, Any]:
        """Analyze demographic distribution from MIMIC records."""
        if records.empty:
            return self._generate_empty_analysis()["demographic_distribution"]

        return {
            "age_distribution": records.groupby(pd.cut(records['age'], bins=[0, 18, 30, 50, 70, 100]))['total_prescriptions'].sum().to_dict(),
            "gender_distribution": records.groupby('gender')['total_prescriptions'].sum().to_dict(),
            "age_specific_outcomes": records.groupby(pd.cut(records['age'], bins=[0, 18, 30, 50, 70, 100]))['readmission_rate'].mean().to_dict()
        }

    def _generate_empty_analysis(self) -> Dict[str, Any]:
        """Generate empty analysis structure when no records are found."""
        return {
            "usage_stats": {
                "total_prescriptions": 0,
                "unique_patients": 0,
                "administration_routes": {},
                "avg_treatment_duration": 0
            },
            "outcomes": {
                "avg_length_of_stay": 0,
                "readmission_rate": 0,
                "common_diagnoses": {}
            },
            "trends": {
                "yearly_usage": {},
                "usage_trend": "No data available"
            },
            "demographics": {
                "age_distribution": {},
                "gender_distribution": {}
            }
        }

    def _calculate_avg_duration(self, records: pd.DataFrame) -> float:
        """Calculate average treatment duration."""
        if 'startdate' in records.columns and 'dischtime' in records.columns:
            return (records['dischtime'] - pd.to_datetime(records['startdate'])).mean().days
        return 0

    def _calculate_readmission_rate(self, records: pd.DataFrame) -> float:
        """Calculate readmission rate."""
        return float(records['readmission_rate'].mean())

    def _calculate_trend(self, records: pd.DataFrame) -> Dict[str, float]:
        """Calculate usage trends over time."""
        try:
            # Group by year and calculate total prescriptions
            yearly_totals = records.groupby('years_in_use')['total_prescriptions'].sum()

            # Calculate year-over-year changes
            trend_data = {}
            for year in range(int(yearly_totals.index.min()), int(yearly_totals.index.max()) + 1):
                if year in yearly_totals.index:
                    trend_data[str(year)] = float(yearly_totals[year])
                else:
                    trend_data[str(year)] = 0.0

            return trend_data

        except Exception as e:
            logger.error(f"Error calculating trends: {str(e)}")
            return {}

    def _determine_trend_direction(self, yearly_usage: Dict[str, float]) -> str:
        """Determine the direction of usage trends."""
        if not yearly_usage:
            return "No data available"

        values = list(yearly_usage.values())
        if len(values) < 2:
            return "Insufficient data"

        # Calculate year-over-year changes
        changes = [values[i] - values[i-1] for i in range(1, len(values))]

        # Calculate percentage of positive/negative changes
        pos_changes = sum(1 for c in changes if c > 0)
        neg_changes = sum(1 for c in changes if c < 0)

        if pos_changes > 0.6 * len(changes):
            return "Increasing"
        elif neg_changes > 0.6 * len(changes):
            return "Decreasing"
        return "Stable"

    def _analyze_seasonal_patterns(self, records: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal patterns in prescription data."""
        try:
            if 'startdate' not in records.columns:
                return {}

            # Convert startdate to datetime if it's not already
            records['startdate'] = pd.to_datetime(records['startdate'])

            # Group by month and calculate average prescriptions
            monthly_avg = records.groupby(records['startdate'].dt.month)['total_prescriptions'].mean()

            # Identify peak months
            peak_months = monthly_avg.nlargest(3)
            low_months = monthly_avg.nsmallest(3)

            return {
                "peak_months": {str(month): float(count) for month, count in peak_months.items()},
                "low_months": {str(month): float(count) for month, count in low_months.items()},
                "seasonality_score": float(monthly_avg.std() / monthly_avg.mean()),
                "has_seasonal_pattern": bool(monthly_avg.std() / monthly_avg.mean() > 0.2)
            }

        except Exception as e:
            logger.error(f"Error analyzing seasonal patterns: {str(e)}")
            return {}

    def _generate_summary(self, records: pd.DataFrame) -> str:
        """Generate a comprehensive summary of medical records analysis."""
        if records.empty:
            return "No medical records data available for analysis"

        total_prescriptions = int(records['total_prescriptions'].sum())
        avg_readmission = float(records['readmission_rate'].mean())
        years_coverage = int(records['years_in_use'].max())

        return f"""
        Medical Records Analysis Summary:
        - Total prescriptions analyzed: {total_prescriptions}
        - Years of data coverage: {years_coverage}
        - Overall readmission rate: {avg_readmission:.2%}
        - Primary admission types: {', '.join(records.groupby('admission_type')['total_prescriptions'].sum().nlargest(3).index.tolist())}
        - Most common discharge locations: {', '.join(records.groupby('discharge_location')['total_prescriptions'].sum().nlargest(3).index.tolist())}
        """
