import logging
from typing import Dict, List, Optional
from ..data_sources.fda_client import FDAClient
from ..config import REPORT_SETTINGS

logger = logging.getLogger(__name__)

class SideEffectsReportGenerator:
    """Generates detailed side effects reports using FDA FAERS data."""

    def __init__(self):
        self.min_confidence = REPORT_SETTINGS['min_confidence_score']
        self.fda_client = None

    async def generate_side_effects_report(
        self,
        smiles: str,
        compound_name: str,
        api_key: str = None,
        description: str = ""
    ) -> Dict:
        """Generate comprehensive side effects report using FDA data."""
        try:
            async with FDAClient(api_key=api_key) as fda_client:
                # Get adverse events data from FDA
                adverse_events = await fda_client.get_adverse_events(compound_name)

                if not adverse_events or not isinstance(adverse_events, dict):
                    logger.warning(f"No valid adverse events found for {compound_name}")
                    return self._empty_report(compound_name)

                # Extract events list from response
                events_list = adverse_events.get('results', [])
                if not events_list:
                    logger.warning(f"No events data in response for {compound_name}")
                    return self._empty_report(compound_name)

                # Analyze event frequency and severity
                event_analysis = self._analyze_events(events_list)

                # Calculate confidence metrics
                confidence_metrics = self._calculate_confidence_metrics(events_list)

                if confidence_metrics['overall_confidence'] < self.min_confidence:
                    logger.warning(f"Low confidence score for {compound_name}")

                return {
                    'compound_name': compound_name,
                    'total_reports': len(events_list),
                    'effects_summary': event_analysis['effects'],
                    'severity_distribution': event_analysis['severity'],
                    'demographic_impact': event_analysis['demographics'],
                    'confidence_metrics': confidence_metrics,
                    'data_sources': ['FDA FAERS'],
                    'report_metadata': {
                        'api_version': 'FDA FAERS API v2.0',
                        'timestamp': adverse_events.get('meta', {}).get('last_updated', '')
                    }
                }

        except Exception as e:
            logger.error(f"Error generating side effects report: {e}", exc_info=True)
            return self._empty_report(compound_name)

    def _analyze_events(self, events: List[Dict]) -> Dict:
        """Analyze adverse events for patterns and severity."""
        try:
            effects = {}
            severity_counts = {'mild': 0, 'moderate': 0, 'severe': 0}
            demographics = {'age_groups': {}, 'gender': {}}

            for event in events:
                # Analyze effect frequency
                reactions = event.get('patient', {}).get('reaction', [])
                for reaction in reactions:
                    effect = reaction.get('reactionmeddrapt', '')
                    if effect:
                        effects[effect] = effects.get(effect, 0) + 1

                # Analyze severity
                severity = self._determine_severity(event)
                severity_counts[severity] += 1

                # Analyze demographics
                self._update_demographics(demographics, event)

            return {
                'effects': [
                    {'effect': k, 'count': v, 'frequency': v/len(events)}
                    for k, v in sorted(effects.items(), key=lambda x: x[1], reverse=True)
                ],
                'severity': severity_counts,
                'demographics': demographics
            }

        except Exception as e:
            logger.error(f"Error analyzing events: {str(e)}")
            return {'effects': [], 'severity': {}, 'demographics': {}}

    def _determine_severity(self, event: Dict) -> str:
        """Determine event severity based on outcome and other factors."""
        try:
            patient_data = event.get('patient', {})
            reactions = patient_data.get('reaction', [])
            if not reactions:
                return 'mild'

            # Check the most severe reaction
            for reaction in reactions:
                outcome = reaction.get('outcome', '').lower()
                if outcome in ['death', 'fatal', '5']:  # FDA uses '5' for death
                    return 'severe'
                elif outcome in ['disability', 'disabling', '2', '6']:  # FDA codes for serious outcomes
                    return 'moderate'

            return 'mild'

        except Exception as e:
            logger.error(f"Error determining severity: {str(e)}")
            return 'moderate'  # Default to moderate if unclear

    def _update_demographics(self, demographics: Dict, event: Dict):
        """Update demographic analysis with event data."""
        try:
            # Update age groups
            age = event.get('patient', {}).get('age')
            if age:
                age_group = self._get_age_group(age)
                demographics['age_groups'][age_group] = demographics['age_groups'].get(age_group, 0) + 1

            # Update gender distribution
            gender = event.get('patient', {}).get('gender', '').lower()
            if gender in ['male', 'female']:
                demographics['gender'][gender] = demographics['gender'].get(gender, 0) + 1

        except Exception as e:
            logger.error(f"Error updating demographics: {str(e)}")

    def _calculate_confidence_metrics(self, events: List[Dict]) -> Dict:
        """Calculate confidence metrics for the side effects report."""
        try:
            total_events = len(events)
            if total_events == 0:
                return {'data_completeness': 0, 'overall_confidence': 0}

            # Calculate completeness of event data
            complete_fields = ['reaction', 'outcome', 'serious', 'patient']
            completeness_scores = []

            for event in events:
                field_score = sum(1 for field in complete_fields if event.get(field)) / len(complete_fields)
                completeness_scores.append(field_score)

            data_completeness = sum(completeness_scores) / len(completeness_scores)

            # Calculate reporting quality
            reporting_quality = sum(1 for event in events
                                 if event.get('outcome') and event.get('patient', {}).get('age')
                                ) / total_events

            # Calculate overall confidence
            confidence = (
                data_completeness * 0.4 +
                reporting_quality * 0.4 +
                min(1.0, total_events / 100) * 0.2  # Scale based on number of reports
            )

            return {
                'data_completeness': round(data_completeness * 100, 2),
                'reporting_quality': round(reporting_quality * 100, 2),
                'total_reports': total_events,
                'overall_confidence': round(confidence * 100, 2)
            }

        except Exception as e:
            logger.error(f"Error calculating confidence metrics: {str(e)}")
            return {
                'data_completeness': 0,
                'reporting_quality': 0,
                'total_reports': 0,
                'overall_confidence': 0
            }

    def _get_age_group(self, age: int) -> str:
        """Determine age group for demographic analysis."""
        if age < 18:
            return 'under_18'
        elif age < 30:
            return '18-29'
        elif age < 50:
            return '30-49'
        elif age < 70:
            return '50-69'
        return '70_plus'

    def _empty_report(self, compound_name: str) -> Dict:
        """Generate an empty report template."""
        return {
            'compound_name': compound_name,
            'total_reports': 0,
            'effects_summary': [],
            'severity_distribution': {},
            'demographic_impact': {},
            'confidence_metrics': {'overall_confidence': 0.0},
            'data_sources': [],
            'report_metadata': {}
        }
