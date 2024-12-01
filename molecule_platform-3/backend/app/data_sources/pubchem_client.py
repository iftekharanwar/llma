import asyncio
import json
import logging
import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple, Union, Any

import aiohttp
from aiohttp import ClientSession, ClientTimeout
from rdkit import Chem

class PubChemClient:
    BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    PUG_URL = f"{BASE_URL}/pug/pug.cgi"
    MAX_RETRIES = 3
    INITIAL_TIMEOUT = 30  # seconds
    BACKOFF_FACTOR = 2
    RATE_LIMIT_DELAY = 2  # seconds between requests
    MIN_REQUEST_INTERVAL = 0.2  # 200ms between requests

    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        """Initialize PubChem client."""
        self.session = session
        self.logger = logging.getLogger(__name__)
        self.last_request_time = 0
        self.timeout = ClientTimeout(total=self.INITIAL_TIMEOUT)

    async def __aenter__(self):
        """Create aiohttp session."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None

    def _validate_and_sanitize_smiles(self, smiles: str) -> Optional[str]:
        """Validate and standardize SMILES string using RDKit."""
        try:
            # Convert SMILES to RDKit mol object
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                self.logger.error(f"Invalid SMILES string: {smiles}")
                return None

            # Remove hydrogens and sanitize
            mol = Chem.RemoveHs(mol)
            Chem.SanitizeMol(mol)

            # Get canonical SMILES
            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

            # Verify the canonical SMILES is valid
            verify_mol = Chem.MolFromSmiles(canonical_smiles)
            if verify_mol is None:
                self.logger.error(f"Failed to validate canonical SMILES: {canonical_smiles}")
                return None

            return canonical_smiles
        except Exception as e:
            self.logger.error(f"Error in SMILES validation: {str(e)}")
            return None

    async def search_similar_compounds(self, smiles: str, similarity_threshold: float = 0.8) -> List[Dict]:
        """Search for similar compounds using multiple methods."""
        try:
            # First get the CID for the input SMILES
            cid = await self._get_cid_from_smiles(smiles)
            if not cid:
                self.logger.error("Failed to get CID for SMILES")
                return []

            self.logger.info(f"Found CID: {cid} for SMILES: {smiles}")

            # Try different similarity search methods
            similar_compounds = []

            # Try fingerprint-based similarity search first
            fingerprint_results = await self._search_similar_by_fingerprint(cid, similarity_threshold)
            if fingerprint_results:
                similar_compounds.extend(fingerprint_results)
                self.logger.info(f"Found {len(fingerprint_results)} compounds via fingerprint search")

            # If we don't have enough results, try 2D similarity search
            if len(similar_compounds) < 5:
                d2_results = await self._search_similar_by_2d(cid, similarity_threshold)
                if d2_results:
                    # Add only new compounds
                    existing_cids = {c['cid'] for c in similar_compounds}
                    new_compounds = [c for c in d2_results if c['cid'] not in existing_cids]
                    similar_compounds.extend(new_compounds)
                    self.logger.info(f"Found {len(new_compounds)} additional compounds via 2D search")

            if not similar_compounds:
                self.logger.warning("No similar compounds found after all attempts")

            return similar_compounds

        except Exception as e:
            self.logger.error(f"Error in similarity search: {str(e)}")
            return []

    async def _get_cid_from_smiles(self, smiles: str, max_attempts: int = 3) -> Optional[int]:
        """Get PubChem CID from SMILES string using multiple fallback methods."""
        try:
            # First standardize the SMILES
            canonical_smiles = self._validate_and_sanitize_smiles(smiles)
            if not canonical_smiles:
                return None

            # URL encode the SMILES string
            encoded_smiles = urllib.parse.quote(canonical_smiles)

            for attempt in range(max_attempts):
                self.logger.info(f"Attempt {attempt + 1}/{max_attempts} - Getting CID for SMILES: {canonical_smiles}")

                # Try different endpoints in sequence with correct URL structure
                endpoints = [
                    f"{self.BASE_URL}/compound/name/{encoded_smiles}/cids/JSON",
                    f"{self.BASE_URL}/compound/smiles/{encoded_smiles}/cids/JSON",
                    f"{self.BASE_URL}/compound/inchi/{encoded_smiles}/cids/JSON"
                ]

                for endpoint in endpoints:
                    try:
                        async with self.session.get(endpoint) as response:
                            if response.status == 200:
                                data = await response.json()
                                if 'IdentifierList' in data and 'CID' in data['IdentifierList']:
                                    cids = data['IdentifierList']['CID']
                                    if cids and len(cids) > 0:
                                        return int(cids[0])
                            elif response.status != 404:  # Log non-404 errors
                                text = await response.text()
                                self.logger.error(f"PubChem API error for {endpoint}: Status {response.status}, Response: {text}")
                    except Exception as e:
                        self.logger.error(f"Error with endpoint {endpoint}: {str(e)}")
                        continue

                # If all endpoints fail, try structure search
                structure_url = f"{self.BASE_URL}/compound/fastsimilarity/smiles/{encoded_smiles}/cids/JSON"
                try:
                    async with self.session.get(structure_url) as response:
                        if response.status == 200:
                            data = await response.json()
                            if 'IdentifierList' in data and 'CID' in data['IdentifierList']:
                                cids = data['IdentifierList']['CID']
                                if cids and len(cids) > 0:
                                    return int(cids[0])
                except Exception as e:
                    self.logger.error(f"Error with structure search: {str(e)}")

                await asyncio.sleep(1)  # Wait before retry

            self.logger.error("Failed to get CID for SMILES after all attempts")
            return None

        except Exception as e:
            self.logger.error(f"Error getting CID from SMILES: {str(e)}")
            return None

    async def find_similar_compounds(self, smiles: str, threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Find similar compounds using PubChem's similarity search."""
        similar_compounds = []

        # First get CID for the input SMILES
        cid = await self._get_cid_from_smiles(smiles)
        if not cid:
            self.logger.warning(f"Could not find CID for SMILES: {smiles}")
            return []

        # Try fingerprint-based similarity search first
        try:
            similar_compounds = await self._search_similar_by_fingerprint(cid, threshold)
            if similar_compounds:
                self.logger.info(f"Found {len(similar_compounds)} similar compounds via fingerprint search")
                return similar_compounds
        except Exception as e:
            self.logger.error(f"Error in fingerprint similarity search: {str(e)}")

        # If fingerprint search fails, try 2D similarity search
        if not similar_compounds:
            try:
                similar_compounds = await self._search_similar_by_2d(cid, threshold)
                if similar_compounds:
                    self.logger.info(f"Found {len(similar_compounds)} similar compounds via 2D search")
                    return similar_compounds
            except Exception as e:
                self.logger.error(f"Error in 2D similarity search: {str(e)}")

        # If both similarity searches fail, try substructure search as last resort
        if not similar_compounds:
            try:
                similar_compounds = await self._search_similar_by_substructure(cid)
                if similar_compounds:
                    self.logger.info(f"Found {len(similar_compounds)} similar compounds via substructure search")
            except Exception as e:
                self.logger.error(f"Error in substructure search: {str(e)}")

        return similar_compounds

    async def _respect_rate_limit(self):
        """Ensure we don't exceed PubChem's rate limit."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.MIN_REQUEST_INTERVAL:
            await asyncio.sleep(self.MIN_REQUEST_INTERVAL - time_since_last)
        self.last_request_time = time.time()

    async def _get_compound_info(self, session: aiohttp.ClientSession, cid: str) -> Optional[Dict]:
        """Get compound information including description and synonyms."""
        try:
            async with session.get(
                f"{self.BASE_URL}/compound/cid/{cid}/description/JSON",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status != 200:
                    return None
                return (await response.json()).get('InformationList', {}).get('Information', [{}])[0]
        except Exception as e:
            self.logger.error(f"Error fetching compound info: {str(e)}")
            return None

    async def _get_bioactivity_data(self, session: aiohttp.ClientSession, cid: str) -> Optional[Dict]:
        """Get bioactivity data for a compound."""
        try:
            async with session.get(
                f"{self.BASE_URL}/compound/cid/{cid}/assaysummary/JSON",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status != 200:
                    return None

                data = await response.json()
                if 'AssaySummary' not in data:
                    return None

                summaries = data.get('AssaySummary', [])
                return {
                    'total_assays': len(summaries),
                    'active_assays': sum(1 for s in summaries if s.get('ActivityOutcome') == 'Active'),
                    'activity_types': list(set(s.get('BioactivityType', '') for s in summaries if s.get('BioactivityType')))
                }
        except Exception as e:
            self.logger.error(f"Error fetching bioactivity data: {str(e)}")
            return None

    def _calculate_data_completeness(self, props: Dict) -> float:
        """Calculate the completeness of compound data as a percentage."""
        required_fields = ['MolecularWeight', 'XLogP', 'TPSA', 'CanonicalSMILES', 'IUPACName', 'Complexity', 'HBondDonorCount', 'HBondAcceptorCount', 'RotatableBondCount']
        available_fields = sum(1 for field in required_fields if props.get(field) is not None)
        return (available_fields / len(required_fields)) * 100

    def _calculate_property_reliability(self, props: Dict) -> float:
        """Calculate the reliability of property data based on completeness and value ranges."""
        try:
            reliability_scores = []

            # Check molecular weight (typical range 0-2000)
            if props.get('MolecularWeight'):
                mw = float(props['MolecularWeight'])
                reliability_scores.append(1.0 if 0 < mw < 2000 else 0.5)

            # Check XLogP (typical range -10 to 10)
            if props.get('XLogP'):
                xlogp = float(props['XLogP'])
                reliability_scores.append(1.0 if -10 <= xlogp <= 10 else 0.5)

            # Check TPSA (typical range 0-200)
            if props.get('TPSA'):
                tpsa = float(props['TPSA'])
                reliability_scores.append(1.0 if 0 <= tpsa <= 200 else 0.5)

            return sum(reliability_scores) / len(reliability_scores) if reliability_scores else 0.0
        except Exception as e:
            self.logger.error(f"Error calculating property reliability: {str(e)}")
            return 0.0

    async def _search_similar_by_fingerprint(self, cid: int, similarity_threshold: float) -> List[Dict]:
        """Search for similar compounds using fingerprint similarity."""
        try:
            # Use PubChem's structure similarity search with correct endpoint
            url = f"{self.BASE_URL}/compound/fastsimilarity_2d/cid/{cid}/property/IUPACName,CanonicalSMILES,MolecularFormula/JSON"
            params = {
                'Threshold': int(similarity_threshold * 100),
                'MaxResults': 100
            }

            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    text = await response.text()
                    self.logger.error(f"Error in fingerprint similarity search: {response.status}, Response: {text}")
                    return []

                data = await response.json()
                if 'PropertyTable' not in data or 'Properties' not in data['PropertyTable']:
                    return []

                compounds = []
                for prop in data['PropertyTable']['Properties']:
                    if prop.get('CID') != cid:  # Exclude the query compound
                        compounds.append({
                            'cid': prop.get('CID'),
                            'name': prop.get('IUPACName', ''),
                            'smiles': prop.get('CanonicalSMILES', ''),
                            'molecular_formula': prop.get('MolecularFormula', ''),
                            'source': 'PubChem'
                        })
                return compounds

        except Exception as e:
            self.logger.error(f"Error in fingerprint similarity search: {str(e)}")
            return []

    async def _search_similar_by_2d(self, smiles: str, similarity_threshold: float) -> List[Dict]:
        """Search similar compounds using PubChem 2D similarity with retry mechanism."""
        max_retries = 3
        base_delay = 2  # Base delay in seconds

        for attempt in range(max_retries):
            try:
                # URL encode the SMILES string
                encoded_smiles = urllib.parse.quote(smiles)

                # First get similar compound CIDs
                search_url = f"{self.BASE_URL}/compound/similarity/smiles/{encoded_smiles}/cids/JSON"
                params = {
                    'Threshold': int(similarity_threshold * 100),
                    'MaxResults': 100
                }

                delay = base_delay * (2 ** attempt)  # Exponential backoff
                self.logger.info(f"Attempt {attempt + 1}/{max_retries} - Waiting {delay}s before retry")
                await asyncio.sleep(delay)

                timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
                async with self.session.get(search_url, params=params, timeout=timeout) as response:
                    if response.status != 200:
                        text = await response.text()
                        self.logger.error(f"2D similarity search failed: {response.status}, Response: {text}")
                        continue

                    data = await response.json()
                    if 'IdentifierList' not in data or 'CID' not in data['IdentifierList']:
                        continue

                    similar_cids = data['IdentifierList']['CID']

                    # Then get properties for the similar compounds
                    if similar_cids:
                        props_url = f"{self.BASE_URL}/compound/cid/{','.join(map(str, similar_cids))}/property/IUPACName,CanonicalSMILES,MolecularFormula/JSON"
                        async with self.session.get(props_url, timeout=timeout) as props_response:
                            if props_response.status == 200:
                                props_data = await props_response.json()
                                if 'PropertyTable' in props_data and 'Properties' in props_data['PropertyTable']:
                                    compounds = []
                                    for prop in props_data['PropertyTable']['Properties']:
                                        compounds.append({
                                            'cid': prop.get('CID'),
                                            'name': prop.get('IUPACName', ''),
                                            'smiles': prop.get('CanonicalSMILES', ''),
                                            'molecular_formula': prop.get('MolecularFormula', ''),
                                            'source': 'PubChem'
                                        })
                                    if compounds:
                                        return compounds

            except asyncio.TimeoutError:
                self.logger.error(f"Timeout during attempt {attempt + 1}")
            except Exception as e:
                self.logger.error(f"Error in attempt {attempt + 1}: {str(e)}")

        self.logger.error("All similarity search attempts failed")
        return []

    async def _search_similar_by_substructure(self, smiles: str, similarity_threshold: float) -> List[Dict]:
        """Search similar compounds using substructure match."""
        try:
            url = f"{self.BASE_URL}/compound/substructure/smiles/{smiles}/JSON"
            params = {'MaxResults': 10}
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    return []
                data = await response.json()
                compounds = await self._process_compound_results(data)
                return compounds[:10]  # Limit results

        except Exception as e:
            self.logger.error(f"Substructure search failed: {str(e)}")
            return []

    async def _process_compound_results(self, compounds: List[Dict]) -> List[Dict]:
        """Process and validate compound results."""
        processed_compounds = []
        for compound in compounds:
            try:
                cid = compound.get('cid')
                if not cid:
                    continue

                # Get detailed compound data
                compound_data = await self._get_compound_data(self.session, str(cid))
                if not compound_data:
                    continue

                # Extract and validate SMILES
                smiles = compound_data.get('smiles')
                if not smiles:
                    continue

                processed_compounds.append({
                    'name': compound_data.get('name', ''),
                    'smiles': smiles,
                    'similarity': compound.get('similarity', 0.0),
                    'molecular_weight': compound_data.get('properties', {}).get('molecular_weight'),
                    'formula': compound_data.get('properties', {}).get('molecular_formula'),
                    'properties': compound_data.get('properties', {})
                })

            except Exception as e:
                self.logger.error(f"Error processing compound result: {str(e)}")
                continue

        return processed_compounds
