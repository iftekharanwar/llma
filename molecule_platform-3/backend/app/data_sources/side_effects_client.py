import aiohttp
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class SideEffectsClient:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the FDA FAERS API client."""
        self.api_key = api_key
        self.base_url = "https://api.fda.gov/drug/event.json"
        self.last_response = None
        self.rate_limit_remaining = None
        self.rate_limit_reset = None
        self._request_count = 0
        self._last_request_time = None
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def set_api_key(self, api_key: str) -> None:
        """Set the FDA API key."""
        self.api_key = api_key

    async def get_side_effects(self, compound_name: str, smiles: str, max_retries: int = 3) -> Dict[str, Any]:
        """Fetch side effects data from FDA's FAERS database with retry logic."""
        if not self.api_key:
            logger.error("FDA API key not set")
            return self._empty_effects_template()

        try:
            for attempt in range(max_retries):
                try:
                    search_query = self._build_search_query(compound_name)
                    logger.debug(f"Attempt {attempt + 1}/{max_retries} for compound {compound_name}")

                    params = {
                        'api_key': self.api_key,
                        'search': search_query,
                        'limit': 100
                    }

                    async with self.session.get(self.base_url, params=params) as response:
                        response_text = await response.text()
                        logger.debug(f"FDA API Response Status: {response.status}")
                        logger.debug(f"FDA API Response: {response_text[:500]}...")  # Log first 500 chars

                        if response.status == 200:
                            try:
                                data = json.loads(response_text)
                                if data.get('results'):
                                    logger.info(f"Found {len(data['results'])} records for {compound_name}")
                                    return self._process_side_effects(data)
                                else:
                                    logger.warning(f"No results found for {compound_name}")
                            except json.JSONDecodeError as je:
                                logger.error(f"JSON decode error: {str(je)}")
                        elif response.status == 404:
                            logger.warning(f"No data found for compound {compound_name}")
                            break
                        else:
                            logger.error(f"FDA API error: {response.status} - {response_text}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2 ** attempt)
                                continue

                except aiohttp.ClientError as ce:
                    logger.error(f"Client error: {str(ce)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue

            return self._empty_effects_template()

        except Exception as e:
            logger.error(f"Error fetching side effects: {str(e)}")
            return self._empty_effects_template()

    def _process_side_effects(self, data: Dict) -> Dict[str, Any]:
        """Process and analyze the side effects data."""
        if not data.get('results'):
            return self._empty_effects_template()

        effects = {}
        severities = {'mild': 0, 'moderate': 0, 'severe': 0}
        outcomes = {}

        for result in data['results']:
            self._extract_effects(result, effects)
            self._extract_severity(result, severities)
            self._extract_outcomes(result, outcomes)

        return {
            "effects_summary": self._summarize_effects(effects),
            "severity_distribution": severities,
            "frequency_analysis": self._analyze_frequency(effects),
            "confidence_metrics": {
                "total_reports": len(data['results']),
                "data_completeness": self._calculate_completeness(data['results']),
                "reporting_quality": self._assess_reporting_quality(data['results']),
                "timestamp": datetime.now().isoformat()
            }
        }

    def _extract_effects(self, result: Dict, effects: Dict) -> None:
        """Extract side effects from a single report."""
        try:
            if 'patient' in result:
                for drug in result['patient'].get('drug', []):
                    if drug.get('medicinalproduct') or drug.get('openfda', {}).get('generic_name'):
                        for reaction in result['patient'].get('reaction', []):
                            effect = reaction.get('reactionmeddrapt')
                            if effect:
                                effects[effect] = effects.get(effect, 0) + 1
        except Exception as e:
            logger.error(f"Error extracting effects: {str(e)}")

    def _extract_severity(self, result: Dict, severities: Dict) -> None:
        """Extract severity information from a single report."""
        if 'serious' in result:
            severity = 'severe' if result['serious'] else 'mild'
            severities[severity] = severities.get(severity, 0) + 1

    def _extract_outcomes(self, result: Dict, outcomes: Dict) -> None:
        """Extract patient outcomes from a single report."""
        if 'patient' in result and 'reaction' in result['patient']:
            for reaction in result['patient']['reaction']:
                if 'outcome' in reaction:
                    outcome = reaction['outcome']
                    outcomes[outcome] = outcomes.get(outcome, 0) + 1

    def _summarize_effects(self, effects: Dict) -> Dict[str, Any]:
        """Summarize the collected side effects data."""
        sorted_effects = sorted(effects.items(), key=lambda x: x[1], reverse=True)
        return {
            "most_common": sorted_effects[:10],
            "total_unique_effects": len(effects),
            "total_reports": sum(effects.values())
        }

    def _analyze_frequency(self, effects: Dict) -> Dict[str, Any]:
        """Analyze the frequency distribution of side effects."""
        total_reports = sum(effects.values())
        frequency_ranges = {
            'very_common': 0,    # >10%
            'common': 0,         # 1-10%
            'uncommon': 0,       # 0.1-1%
            'rare': 0,          # <0.1%
        }

        for count in effects.values():
            frequency = (count / total_reports) * 100
            if frequency > 10:
                frequency_ranges['very_common'] += 1
            elif frequency > 1:
                frequency_ranges['common'] += 1
            elif frequency > 0.1:
                frequency_ranges['uncommon'] += 1
            else:
                frequency_ranges['rare'] += 1

        return frequency_ranges

    def _calculate_completeness(self, results: List[Dict]) -> float:
        """Calculate data completeness score."""
        required_fields = ['patient', 'reaction', 'serious']
        scores = []

        for result in results:
            present_fields = sum(1 for field in required_fields if field in result)
            scores.append(present_fields / len(required_fields))

        return sum(scores) / len(scores) if scores else 0

    def _assess_reporting_quality(self, results: List[Dict]) -> float:
        """Assess the quality of the reporting."""
        quality_scores = []

        for result in results:
            score = 0
            if 'patient' in result and 'reaction' in result['patient']:
                reactions = result['patient']['reaction']
                for reaction in reactions:
                    if 'reactionmeddrapt' in reaction:
                        score += 0.5
                    if 'outcome' in reaction:
                        score += 0.5
            quality_scores.append(min(score, 1.0))

        return sum(quality_scores) / len(quality_scores) if quality_scores else 0

    def _empty_effects_template(self) -> Dict[str, Any]:
        """Return empty template when no side effects data found."""
        return {
            "effects_summary": {
                "most_common": [],
                "total_unique_effects": 0,
                "total_reports": 0
            },
            "severity_distribution": {
                "mild": 0,
                "moderate": 0,
                "severe": 0
            },
            "frequency_analysis": {
                "very_common": 0,
                "common": 0,
                "uncommon": 0,
                "rare": 0
            },
            "confidence_metrics": {
                "total_reports": 0,
                "data_completeness": 0,
                "reporting_quality": 0
            }
        }

    def _should_wait(self) -> bool:
        """Determine if we should wait due to rate limiting."""
        if self.rate_limit_remaining is not None and self.rate_limit_remaining <= 1:
            return True
        if self._last_request_time and (datetime.now() - self._last_request_time).total_seconds() < 1:
            return True
        return False

    def _calculate_wait_time(self) -> int:
        """Calculate wait time based on rate limit reset."""
        if self.rate_limit_reset:
            return max(0, (self.rate_limit_reset - datetime.now()).total_seconds())
        return 1

    def _update_rate_limits(self, response: aiohttp.ClientResponse) -> None:
        """Update rate limit information from response headers."""
        self.rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 1))
        reset_time = response.headers.get('X-RateLimit-Reset')
        if reset_time:
            self.rate_limit_reset = datetime.fromtimestamp(int(reset_time))
        self._last_request_time = datetime.now()

    def _build_search_query(self, compound_name: str) -> str:
        """Build an optimized search query for the FDA API."""
        # Normalize compound name and create variations
        name_variations = [
            compound_name.upper(),
            compound_name.lower(),
            compound_name.title(),
            compound_name.replace(" ", "+"),  # Handle spaces in names
            compound_name.replace("-", " ")   # Handle hyphenated names
        ]

        # Build query with exact and fuzzy matching for all relevant fields
        query_parts = []
        for name in name_variations:
            query_parts.extend([
                f'patient.drug.medicinalproduct:"{name}"',
                f'patient.drug.openfda.generic_name:"{name}"',
                f'patient.drug.openfda.brand_name:"{name}"',
                f'patient.drug.openfda.substance_name:"{name}"'
            ])

        # Combine with OR operator and add existence checks
        return f"({' OR '.join(query_parts)}) AND _exists_:patient.reaction.reactionmeddrapt AND _exists_:patient.drug.medicinalproduct"
