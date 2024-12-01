import aiohttp
import asyncio
from typing import Dict, List, Optional
import logging
import urllib.parse
from ..config import CHEMBL_BASE_URL

logger = logging.getLogger(__name__)

class ChEMBLClient:
    def __init__(self):
        self.base_url = "https://www.ebi.ac.uk/chembl/api/data"
        self.headers = {
            'Accept': 'application/json',
            'User-Agent': 'SynthMolAssistant/1.0'
        }
        self.session = None
        self.timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self._circuit_breaker_fails = 0
        self._max_fails = 3
        self._reset_time = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers=self.headers, timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session and not self.session.closed:
            await self.session.close()

    async def search_similar_compounds(self, smiles: str, similarity_threshold: float = 0.7) -> List[Dict]:
        """Search for compounds similar to the input SMILES string."""
        try:
            if self._circuit_breaker_fails >= self._max_fails:
                if self._reset_time and (asyncio.get_event_loop().time() - self._reset_time) < 300:
                    logger.warning("Circuit breaker active, skipping ChEMBL")
                    return []
                self._circuit_breaker_fails = 0
                self._reset_time = None

            if not self.session:
                raise RuntimeError("ChEMBLClient must be used as an async context manager")

            encoded_smiles = urllib.parse.quote(smiles)
            url = f"{self.base_url}/{encoded_smiles}/{int(similarity_threshold * 100)}"
            params = {
                'limit': 10
            }

            # Add detailed request logging
            logger.info(f"Making ChEMBL similarity search request:")
            logger.info(f"Base URL: {self.base_url}")
            logger.info(f"URL: {url}")
            logger.info(f"Parameters: {params}")
            max_retries = 3
            retry_delay = 1

            for attempt in range(max_retries):
                try:
                    async with self.session.get(url, params=params) as response:
                        if response.status == 429:  # Rate limit exceeded
                            logger.warning("ChEMBL rate limit exceeded, retrying...")
                            await asyncio.sleep(retry_delay * (2 ** attempt))
                            continue

                        if response.status != 200:
                            response_text = await response.text()
                            logger.error(f"ChEMBL API error: Status {response.status}, Response: {response_text}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(retry_delay * (2 ** attempt))
                                continue
                            self._circuit_breaker_fails += 1
                            return []

                        data = await response.json()
                        logger.debug(f"ChEMBL API Response: {data}")

                        molecules = data.get('molecules', [])
                        if not molecules:
                            logger.warning("No similar compounds found in ChEMBL")
                            return []

                        # Process compounds in parallel with rate limiting
                        tasks = []
                        for molecule in molecules[:10]:
                            tasks.append(self._enrich_compound_data(self.session, molecule))
                            await asyncio.sleep(0.2)  # Rate limiting

                        compound_data_list = await asyncio.gather(*tasks, return_exceptions=True)
                        enriched_compounds = [data for data in compound_data_list if isinstance(data, dict)]

                        if enriched_compounds:
                            self._circuit_breaker_fails = 0  # Reset on success
                            return sorted(enriched_compounds, key=lambda x: x['similarity'], reverse=True)
                        return []

                except aiohttp.ClientError as ce:
                    logger.error(f"Client error in ChEMBL API: {str(ce)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (2 ** attempt))
                        continue
                    self._circuit_breaker_fails += 1
                    return []

        except Exception as e:
            logger.error(f"Error fetching similar compounds from ChEMBL: {str(e)}", exc_info=True)
            self._circuit_breaker_fails += 1
            return []

    async def _enrich_compound_data(self, session: aiohttp.ClientSession, molecule: Dict) -> Optional[Dict]:
        """Enrich compound data with detailed properties and additional information."""
        try:
            properties = await self._get_compound_properties(session, molecule.get('molecule_chembl_id'))
            if not properties:
                return None

            # Get bioactivity data if available
            bioactivity = await self._get_bioactivity_data(session, molecule.get('molecule_chembl_id'))

            return {
                'chembl_id': molecule.get('molecule_chembl_id'),
                'smiles': molecule.get('molecule_structures', {}).get('canonical_smiles'),
                'similarity': molecule.get('similarity'),
                'name': molecule.get('pref_name'),
                'source': 'ChEMBL',
                'properties': properties,
                'bioactivity': bioactivity,
                'references': {
                    'chembl_url': f"https://www.ebi.ac.uk/chembl/compound_report_card/{molecule.get('molecule_chembl_id')}",
                    'pubmed_ids': molecule.get('cross_references', {}).get('pubmed_ids', [])
                },
                'confidence_metrics': {
                    'data_completeness': properties.get('data_completeness', 0),
                    'has_bioactivity': 1.0 if bioactivity else 0.0,
                    'has_references': 1.0 if molecule.get('cross_references', {}).get('pubmed_ids') else 0.0
                }
            }
        except Exception as e:
            logger.error(f"Error enriching compound data: {str(e)}")
            return None

    async def _get_compound_properties(self, session: aiohttp.ClientSession, chembl_id: str) -> Optional[Dict]:
        """Get detailed properties for a specific compound."""
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                async with session.get(f"{self.base_url}/molecule/{chembl_id}") as response:
                    if response.status == 429:  # Rate limit
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay * (2 ** attempt))
                            continue
                        return None

                    if response.status != 200:
                        logger.error(f"Error fetching compound {chembl_id}: Status {response.status}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay * (2 ** attempt))
                            continue
                        return None

                    data = await response.json()

                    if not data.get('molecule_properties'):
                        logger.warning(f"No properties found for compound {chembl_id}")
                        return None

                    properties = {
                        'molecular_weight': data['molecule_properties'].get('full_mwt'),
                        'alogp': data['molecule_properties'].get('alogp'),
                        'psa': data['molecule_properties'].get('psa'),
                        'qed_weighted': data['molecule_properties'].get('qed_weighted'),
                        'ro5_violations': data['molecule_properties'].get('num_ro5_violations'),
                        'data_completeness': self._calculate_data_completeness(data)
                    }

                    # Only return if we have at least molecular weight and one other property
                    if properties['molecular_weight'] and any(v for k, v in properties.items()
                                                           if k not in ['molecular_weight', 'data_completeness']):
                        return properties
                    return None

            except aiohttp.ClientError as ce:
                logger.error(f"Client error fetching compound {chembl_id}: {str(ce)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                    continue
                return None
            except Exception as e:
                logger.error(f"Error fetching compound {chembl_id}: {str(e)}")
                return None

        return None

    async def _get_bioactivity_data(self, session: aiohttp.ClientSession, chembl_id: str) -> Optional[Dict]:
        """Get bioactivity data for a compound."""
        try:
            url = f"{self.base_url}/mechanism/{chembl_id}"
            async with session.get(url) as response:
                if response.status != 200:
                    return None

                data = await response.json()
                mechanisms = data.get('mechanisms', [])

                if not mechanisms:
                    return None

                return {
                    'action_type': [m.get('action_type') for m in mechanisms if m.get('action_type')],
                    'target_type': [m.get('target_type') for m in mechanisms if m.get('target_type')],
                    'mechanism_of_action': [m.get('mechanism_of_action') for m in mechanisms if m.get('mechanism_of_action')]
                }
        except Exception as e:
            logger.error(f"Error fetching bioactivity data: {str(e)}")
            return None

    def _calculate_data_completeness(self, compound_data: Dict) -> float:
        """Calculate the completeness of compound data as a percentage."""
        required_fields = [
            'full_mwt', 'alogp', 'psa', 'qed_weighted',
            'num_ro5_violations', 'aromatic_rings'
        ]

        if 'molecule_properties' not in compound_data:
            return 0.0

        props = compound_data['molecule_properties']
        available_fields = sum(1 for field in required_fields if props.get(field) is not None)
        return (available_fields / len(required_fields)) * 100
