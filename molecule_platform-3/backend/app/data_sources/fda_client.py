import aiohttp
import asyncio
import logging
from typing import Dict, List, Any, Optional
from ..config import FDA_API_KEY

logger = logging.getLogger(__name__)

class FDAClient:
    """Client for interacting with FDA FAERS API."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or FDA_API_KEY
        self.base_url = "https://api.fda.gov/drug"
        self.session = None
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        self._rate_limit_remaining = 240  # FDA default rate limit
        self._rate_limit_reset = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_adverse_events(self, drug_name: str, limit: int = 100) -> Optional[Dict[str, Any]]:
        """Get adverse events data from FDA FAERS."""
        try:
            url = f"{self.base_url}/event.json"
            # Use exact match for drug name and include brand names
            params = {
                'api_key': self.api_key,
                'search': f'(patient.drug.medicinalproduct:"{drug_name}" OR patient.drug.openfda.brand_name.exact:"{drug_name}" OR patient.drug.openfda.generic_name.exact:"{drug_name}") AND _exists_:patient.reaction',
                'limit': limit
            }

            for attempt in range(self.max_retries):
                try:
                    if self._rate_limit_remaining <= 0:
                        await asyncio.sleep(5)  # Wait for rate limit reset

                    async with self.session.get(url, params=params) as response:
                        # Update rate limit info
                        self._rate_limit_remaining = int(response.headers.get('x-ratelimit-remaining', 240))

                        if response.status == 200:
                            data = await response.json()
                            logger.info(f"FDA API Response for {drug_name}: Status 200")
                            results = data.get('results', [])
                            logger.info(f"Found {len(results)} event records")
                            if not results:
                                logger.warning(f"No adverse events found for {drug_name}")
                            return results
                        elif response.status == 429:  # Rate limit exceeded
                            logger.warning("FDA API rate limit exceeded")
                            await asyncio.sleep(self.retry_delay * (attempt + 1))
                        elif response.status == 404:
                            logger.warning(f"No adverse events found for drug: {drug_name}")
                            return None
                        else:
                            logger.error(f"Error fetching adverse events: {response.status}")
                            await asyncio.sleep(self.retry_delay * (attempt + 1))
                except aiohttp.ClientError as e:
                    logger.error(f"Client error in FDA FAERS API call: {str(e)}")
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

            return None
        except Exception as e:
            logger.error(f"Error in FDA FAERS API call: {str(e)}")
            return None

    async def get_drug_label(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """Get drug label information from FDA."""
        try:
            url = f"{self.base_url}/label.json"
            params = {
                'api_key': self.api_key,
                'search': f'openfda.generic_name:"{drug_name}" OR openfda.brand_name:"{drug_name}"',
                'limit': 1
            }

            for attempt in range(self.max_retries):
                try:
                    if self._rate_limit_remaining <= 0:
                        await asyncio.sleep(5)  # Wait for rate limit reset

                    async with self.session.get(url, params=params) as response:
                        # Update rate limit info
                        self._rate_limit_remaining = int(response.headers.get('x-ratelimit-remaining', 240))

                        if response.status == 200:
                            data = await response.json()
                            return self._process_drug_label(data)
                        elif response.status == 429:  # Rate limit exceeded
                            logger.warning("FDA API rate limit exceeded")
                            await asyncio.sleep(self.retry_delay * (attempt + 1))
                        elif response.status == 404:
                            logger.warning(f"No drug label found for: {drug_name}")
                            return None
                        else:
                            logger.error(f"Error fetching drug label: {response.status}")
                            await asyncio.sleep(self.retry_delay * (attempt + 1))
                except aiohttp.ClientError as e:
                    logger.error(f"Client error in FDA label API call: {str(e)}")
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

            return None
        except Exception as e:
            logger.error(f"Error in FDA label API call: {str(e)}")
            return None

    def _process_adverse_events(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process adverse events data from FDA FAERS."""
        try:
            results = data.get('results', [])
            events_summary = self._summarize_events(results)
            severity_distribution = self._analyze_severity(results)
            demographic_data = self._analyze_demographics(results)

            return {
                'total_reports': len(results),
                'events_summary': events_summary,
                'severity_distribution': severity_distribution,
                'demographic_data': demographic_data,
                'confidence_metrics': self._calculate_confidence_metrics(results)
            }
        except Exception as e:
            logger.error(f"Error processing adverse events: {str(e)}")
            return {}

    def _summarize_events(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize adverse events data."""
        event_counts = {}
        total_events = 0

        for result in results:
            reactions = result.get('patient', {}).get('reaction', [])
            for reaction in reactions:
                event = reaction.get('reactionmeddrapt', '').lower()
                if event:
                    event_counts[event] = event_counts.get(event, 0) + 1
                    total_events += 1

        # Sort events by frequency
        sorted_events = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)

        return {
            'total_events': total_events,
            'unique_events': len(event_counts),
            'most_common': sorted_events[:10],
            'event_frequency': {
                event: count/total_events
                for event, count in event_counts.items()
            } if total_events > 0 else {}
        }

    def _analyze_severity(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze severity of adverse events."""
        severity_counts = {
            'mild': 0,
            'moderate': 0,
            'severe': 0
        }

        for result in results:
            if result.get('serious'):
                severity_counts['severe'] += 1
            else:
                # Use outcome to determine mild vs moderate
                outcome = result.get('patient', {}).get('reaction', [{}])[0].get('outcome')
                if outcome in ['recovered', 'recovering']:
                    severity_counts['mild'] += 1
                else:
                    severity_counts['moderate'] += 1

        return severity_counts

    def _analyze_demographics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze demographic information from adverse events."""
        age_groups = {
            '0-18': 0,
            '19-30': 0,
            '31-50': 0,
            '51-70': 0,
            '70+': 0
        }
        gender_counts = {'male': 0, 'female': 0, 'unknown': 0}

        for result in results:
            patient = result.get('patient', {})

            # Process age
            age = patient.get('patientonsetage')
            if age:
                if age <= 18:
                    age_groups['0-18'] += 1
                elif age <= 30:
                    age_groups['19-30'] += 1
                elif age <= 50:
                    age_groups['31-50'] += 1
                elif age <= 70:
                    age_groups['51-70'] += 1
                else:
                    age_groups['70+'] += 1

            # Process gender
            gender = patient.get('patientsex', '').lower()
            if gender == '1':
                gender_counts['male'] += 1
            elif gender == '2':
                gender_counts['female'] += 1
            else:
                gender_counts['unknown'] += 1

        return {
            'age_distribution': age_groups,
            'gender_distribution': gender_counts
        }

    def _calculate_confidence_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate confidence metrics for the data."""
        total_reports = len(results)
        if total_reports == 0:
            return {
                'data_completeness': 0,
                'reporting_quality': 0,
                'sample_size_confidence': 0
            }

        # Calculate data completeness
        required_fields = ['patient.reaction', 'patient.drug', 'serious']
        completeness_scores = []

        for result in results:
            score = 0
            if result.get('patient', {}).get('reaction'):
                score += 1
            if result.get('patient', {}).get('drug'):
                score += 1
            if 'serious' in result:
                score += 1
            completeness_scores.append(score / len(required_fields))

        data_completeness = sum(completeness_scores) / len(completeness_scores)

        # Calculate reporting quality based on detail level
        quality_scores = []
        for result in results:
            score = 0
            if result.get('patient', {}).get('patientonsetage'):
                score += 0.2
            if result.get('patient', {}).get('patientsex'):
                score += 0.2
            if result.get('serious'):
                score += 0.2
            if result.get('patient', {}).get('reaction', [{}])[0].get('outcome'):
                score += 0.2
            if result.get('patient', {}).get('drug', [{}])[0].get('drugcharacterization'):
                score += 0.2
            quality_scores.append(score)

        reporting_quality = sum(quality_scores) / len(quality_scores)

        # Calculate sample size confidence
        sample_size_confidence = min(total_reports / 1000, 1.0)  # Normalize to max of 1.0

        return {
            'data_completeness': round(data_completeness, 2),
            'reporting_quality': round(reporting_quality, 2),
            'sample_size_confidence': round(sample_size_confidence, 2)
        }

    def _process_drug_label(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process drug label data from FDA."""
        try:
            if not data.get('results'):
                return {}

            label = data['results'][0]
            return {
                'warnings': label.get('warnings', []),
                'adverse_reactions': label.get('adverse_reactions', []),
                'drug_interactions': label.get('drug_interactions', []),
                'contraindications': label.get('contraindications', []),
                'boxed_warnings': label.get('boxed_warnings', [])
            }
        except Exception as e:
            logger.error(f"Error processing drug label: {str(e)}")
            return {}
