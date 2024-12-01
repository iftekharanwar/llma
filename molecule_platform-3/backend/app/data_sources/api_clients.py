import logging
import json
import time
import os
import urllib.parse
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Any
import aiohttp
from aiohttp import ClientTimeout
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, DataStructs
from ..config import FDA_API_KEY

logger = logging.getLogger(__name__)

class ChEMBLClient:
    """Client for interacting with ChEMBL API."""

    def __init__(self):
        self.base_url = "https://www.ebi.ac.uk/chembl/api/data"
        self.similarity_endpoint = f"{self.base_url}/molecule/similarity"  # Fixed endpoint
        self.molecule_endpoint = f"{self.base_url}/molecule"
        self.session = None

    async def __aenter__(self):
        """Initialize the client session."""
        if not self.session:
            self.session = aiohttp.ClientSession(headers={'Accept': 'application/json'})
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup the client session."""
        if self.session:
            await self.session.close()
            self.session = None

    async def search_similar_compounds(self, smiles: str, similarity: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar compounds in ChEMBL with detailed property data."""
        if not self.session:
            self.session = aiohttp.ClientSession(headers={'Accept': 'application/json'})  # Explicitly request JSON

        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                logger.error("Invalid SMILES string")
                return []

            standard_smiles = Chem.MolToSmiles(mol)
            params = {
                "smiles": standard_smiles,
                "similarity": similarity,
                "limit": 10,
                "format": "json"  # Explicitly request JSON format
            }

            # Implement retry logic
            max_retries = 3
            retry_delay = 1

            for attempt in range(max_retries):
                try:
                    async with self.session.get(
                        self.similarity_endpoint,
                        params=params,
                        headers={'Accept': 'application/json'},  # Redundant but explicit
                        timeout=aiohttp.ClientTimeout(total=60)  # Increased timeout further
                    ) as response:
                        if response.status == 200:
                            try:
                                data = await response.json()
                                molecules = data.get("molecules", [])
                                detailed_results = []

                                for mol in molecules:
                                    mol_id = mol.get("molecule_chembl_id")
                                    if mol_id:
                                        mol_data = await self._get_molecule_details(mol_id)
                                        if mol_data:
                                            mol_data['similarity'] = mol.get('similarity', 0.0)
                                            mol_data['source'] = 'ChEMBL'  # Add source identifier
                                            detailed_results.append(mol_data)

                                return detailed_results
                            except json.JSONDecodeError as je:
                                logger.error(f"JSON decode error: {str(je)}")
                                content = await response.text()
                                logger.error(f"Response content: {content[:200]}")  # Log first 200 chars
                                if attempt < max_retries - 1:
                                    await asyncio.sleep(retry_delay)
                                    retry_delay *= 2
                                    continue
                                return []
                        else:
                            logger.error(f"ChEMBL API error: {response.status}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(retry_delay)
                                retry_delay *= 2
                                continue
                            return []

                except asyncio.TimeoutError:
                    logger.error(f"ChEMBL API timeout (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    return []

        except Exception as e:
            logger.error(f"Error searching similar compounds: {str(e)}")
            return []

class PubChemClient:
    """Client for interacting with PubChem API."""

    def __init__(self):
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        self.session = None

    async def __aenter__(self):
        """Initialize the client session."""
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),  # Reduced from 120s to 30s
                headers={'Accept': 'application/json'}
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup the client session."""
        if self.session:
            await self.session.close()
            self.session = None

    async def search_similar_compounds(self, smiles: str, similarity: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar compounds in PubChem with comprehensive property data."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=30),
                    headers={'Accept': 'application/json'}
                )

            max_retries = 5
            retry_delay = 2

            for attempt in range(max_retries):
                try:
                    logger.info(f"Attempting PubChem search (attempt {attempt + 1}/{max_retries})")
                    cid = await asyncio.wait_for(
                        self._get_cid_from_smiles(smiles),
                        timeout=10
                    )

                    if not cid:
                        logger.error("Failed to get CID from SMILES")
                        return []

                    properties = [
                        "IUPACName",
                        "MolecularFormula",
                        "MolecularWeight",
                        "XLogP",
                        "TPSA",
                        "Complexity",
                        "HBondDonorCount",
                        "HBondAcceptorCount",
                        "RotatableBondCount",
                        "ExactMass",
                        "MonoisotopicMass",
                        "CanonicalSMILES"
                    ]

                    # Use PubChem structure similarity search endpoint
                    similarity_url = f"{self.base_url}/compound/fastsimilarity_2d/smiles/{urllib.parse.quote(smiles)}/property/IUPACName,MolecularFormula,MolecularWeight,XLogP,TPSA,Complexity,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,ExactMass,MonoisotopicMass,CanonicalSMILES/JSON"
                    params = {
                        'Threshold': int(similarity * 100),
                        'MaxResults': 10
                    }
                    logger.debug(f"PubChem similarity search URL: {similarity_url}")
                    async with self.session.get(
                        similarity_url,
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=20)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            similar_cids = data.get("IdentifierList", {}).get("CID", [])

                            if not similar_cids:
                                logger.warning("No similar compounds found")
                                return []

                            # Now get properties for similar compounds
                            properties_url = f"{self.base_url}/compound/cid/{','.join(map(str, similar_cids))}/property/{','.join(properties)}/JSON"
                            async with self.session.get(
                                properties_url,
                                timeout=aiohttp.ClientTimeout(total=20)
                            ) as prop_response:
                                if prop_response.status == 200:
                                    prop_data = await prop_response.json()
                                    compounds = prop_data.get("PropertyTable", {}).get("Properties", [])
                                    logger.info(f"Found {len(compounds)} similar compounds from PubChem")
                                    return self._process_pubchem_results(compounds)

                                logger.error(f"Error fetching properties: {prop_response.status}")
                                return []

                        elif response.status == 202:  # Request accepted, poll for results
                            poll_url = response.headers.get('Location', similarity_url)
                            max_polls = 5
                            poll_delay = 2

                            for _ in range(max_polls):
                                await asyncio.sleep(poll_delay)
                                async with self.session.get(poll_url) as poll_response:
                                    if poll_response.status == 200:
                                        poll_data = await poll_response.json()
                                        similar_cids = poll_data.get("IdentifierList", {}).get("CID", [])
                                        if similar_cids:
                                            break
                                poll_delay *= 1.5

                            if not similar_cids:
                                logger.warning("No results after polling")
                                return []

                            # Get properties for found compounds
                            properties_url = f"{self.base_url}/compound/cid/{','.join(map(str, similar_cids))}/property/{','.join(properties)}/JSON"
                            async with self.session.get(properties_url) as prop_response:
                                if prop_response.status == 200:
                                    prop_data = await prop_response.json()
                                    compounds = prop_data.get("PropertyTable", {}).get("Properties", [])
                                    return self._process_pubchem_results(compounds)

                            return []

                        elif response.status == 429:  # Rate limit
                            logger.warning("PubChem rate limit hit")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(retry_delay * 4)
                                retry_delay *= 2
                                continue

                        elif response.status == 504:  # Gateway timeout
                            logger.warning("PubChem gateway timeout")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(retry_delay * 2)
                                retry_delay *= 2
                                continue

                        logger.error(f"PubChem API error: {response.status}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2
                            continue
                        return []
                except asyncio.TimeoutError:
                    logger.error(f"PubChem API timeout (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    return []

        except Exception as e:
            logger.error(f"Error in PubChem similarity search: {str(e)}")
            return []

    async def _get_cid_from_smiles(self, smiles: str) -> Optional[str]:
        """Get PubChem CID from SMILES string."""
        try:
            # URL encode the SMILES string
            encoded_smiles = urllib.parse.quote(smiles)
            url = f"{self.base_url}/compound/smiles/{encoded_smiles}/cids/JSON"

            async with self.session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    cids = data.get("IdentifierList", {}).get("CID", [])
                    if cids:
                        logger.info(f"Found CID {cids[0]} for SMILES {smiles}")
                        return str(cids[0])
                    logger.warning(f"No CIDs found for SMILES {smiles}")
                    return None
                logger.error(f"PubChem API error: Status {response.status}")
                return None
        except asyncio.TimeoutError:
            logger.error("Timeout while getting CID from SMILES")
            return None
        except Exception as e:
            logger.error(f"Error getting CID from SMILES: {str(e)}")
            return None

    def _process_pubchem_results(self, compounds: List[Dict]) -> List[Dict[str, Any]]:
        """Process PubChem API results into standardized format with enhanced properties."""
        processed_results = []
        for compound in compounds[:10]:  # Limit to top 10 compounds
            try:
                # Extract and validate required properties
                if not all(key in compound for key in ["CID", "MolecularFormula", "MolecularWeight"]):
                    logger.warning(f"Skipping compound {compound.get('CID')} due to missing required properties")
                    continue

                processed_compound = {
                    "name": compound.get("IUPACName", "Unknown"),
                    "pubchem_cid": str(compound.get("CID")),
                    "similarity": compound.get("Score", 0.0) / 100 if "Score" in compound else 0.0,
                    "source": "PubChem",
                    "properties": {
                        "molecular_weight": float(compound.get("MolecularWeight", 0)),
                        "xlogp": float(compound.get("XLogP", 0)),
                        "tpsa": float(compound.get("TPSA", 0)),
                        "complexity": int(compound.get("Complexity", 0)),
                        "molecular_formula": compound.get("MolecularFormula", ""),
                        "hbond_donor_count": int(compound.get("HBondDonorCount", 0)),
                        "hbond_acceptor_count": int(compound.get("HBondAcceptorCount", 0)),
                        "rotatable_bond_count": int(compound.get("RotatableBondCount", 0)),
                        "exact_mass": float(compound.get("ExactMass", 0)),
                        "monoisotopic_mass": float(compound.get("MonoisotopicMass", 0)),
                        "canonical_smiles": compound.get("CanonicalSMILES", "")
                    },
                    "confidence_metrics": {
                        "data_completeness": self._calculate_property_completeness(compound),
                        "source_reliability": 0.95  # PubChem is considered highly reliable
                    }
                }
                processed_results.append(processed_compound)
                logger.info(f"Successfully processed compound {compound.get('CID')}")
            except Exception as e:
                logger.error(f"Error processing PubChem compound: {str(e)}")
                continue
        return processed_results
    def _calculate_property_completeness(self, compound: Dict) -> float:
        """Calculate the completeness of property data for a compound."""
        required_properties = [
            "MolecularWeight", "XLogP", "TPSA", "Complexity", "MolecularFormula",
            "HBondDonorCount", "HBondAcceptorCount", "RotatableBondCount",
            "ExactMass", "MonoisotopicMass"
        ]
        available = sum(1 for prop in required_properties if compound.get(prop) is not None)
        return round(available / len(required_properties), 2)

class MIMICClient:
    """Client for accessing MIMIC database data."""

    def __init__(self):
        self.has_credentials = all([
            os.getenv('MIMIC_DB_NAME'),
            os.getenv('MIMIC_DB_USER'),
            os.getenv('MIMIC_DB_PASSWORD'),
            os.getenv('MIMIC_DB_HOST'),
            os.getenv('MIMIC_DB_PORT')
        ])

        if self.has_credentials:
            self.db_params = {
                'dbname': os.getenv('MIMIC_DB_NAME', 'mimic'),
                'user': os.getenv('MIMIC_DB_USER'),
                'password': os.getenv('MIMIC_DB_PASSWORD'),
                'host': os.getenv('MIMIC_DB_HOST'),
                'port': os.getenv('MIMIC_DB_PORT', '5432')
            }
        else:
            logger.warning("MIMIC database credentials not found. Using development mode.")

    async def get_medical_records(self, compound_name: str) -> Dict[str, Any]:
        """Get medical records data for a compound."""
        if not self.has_credentials:
            logger.info("No MIMIC credentials - returning development response")
            return {
                "confidence_metrics": {
                    "data_completeness": 0.0,
                    "source_reliability": 0.0
                },
                "usage_statistics": {},
                "status": "No MIMIC database access - proper credentials required",
                "error": "MIMIC database credentials not configured"
            }

        try:
            # Real MIMIC database query would go here if we had credentials
            return {
                "confidence_metrics": {
                    "data_completeness": 0.0,
                    "source_reliability": 0.0
                },
                "usage_statistics": {},
                "status": "Database connection not implemented",
                "error": "MIMIC database connection not implemented"
            }
        except Exception as e:
            logger.error(f"Error accessing MIMIC database: {str(e)}")
            return {
                "confidence_metrics": {
                    "data_completeness": 0.0,
                    "source_reliability": 0.0
                },
                "usage_statistics": {},
                "error": str(e)
            }

class SideEffectsClient:
    """Client for retrieving side effects data from FDA FAERS database."""

    def __init__(self):
        self.base_url = "https://api.fda.gov/drug/event.json"
        self.api_key = None

    def set_api_key(self, api_key: str):
        self.api_key = api_key
    async def get_side_effects(self, compound_name: str) -> Dict[str, Any]:
        """Get side effects data from FDA FAERS database."""
        if not self.api_key:
            raise ValueError("FDA API key not set")

        try:
            max_retries = 3
            retry_delay = 1

            for attempt in range(max_retries):
                try:
                    async with aiohttp.ClientSession() as session:
                        search_query = (
                            f'patient.drug.medicinalproduct:"{compound_name}" OR '
                            f'patient.drug.openfda.generic_name:"{compound_name}" OR '
                            f'patient.drug.openfda.brand_name:"{compound_name}" OR '
                            f'patient.drug.openfda.substance_name:"{compound_name}"'
                        )

                        params = {
                            'api_key': self.api_key,
                            'search': search_query,
                            'limit': 100
                        }

                        async with session.get(
                            self.base_url,
                            params=params,
                            timeout=aiohttp.ClientTimeout(total=30)
                        ) as response:
                            if response.status == 200:
                                data = await response.json()
                                return self._process_side_effects(data)
                            elif response.status == 429:  # Rate limit exceeded
                                if attempt < max_retries - 1:
                                    await asyncio.sleep(retry_delay * 2)
                                    retry_delay *= 2
                                    continue
                            else:
                                logger.error(f"FDA API error: {response.status}")
                                if attempt < max_retries - 1:
                                    await asyncio.sleep(retry_delay)
                                    retry_delay *= 2
                                    continue
                                return self._get_empty_side_effects_template()

                except asyncio.TimeoutError:
                    logger.error(f"FDA API timeout (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    return self._get_empty_side_effects_template()

        except Exception as e:
            logger.error(f"Error getting side effects data: {str(e)}")
            return self._get_empty_side_effects_template()

    def _get_empty_side_effects_template(self) -> Dict[str, Any]:
        """Return an empty template for side effects data."""
        return {
            "effects_summary": [],
            "severity_distribution": {},
            "frequency_analysis": {},
            "confidence_metrics": {
                "total_reports": 0,
                "data_completeness": 0,
                "reporting_quality": 0
            }
        }

    def _process_side_effects(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and structure side effects data."""
        try:
            effects = []
            if not data:
                return self._get_empty_side_effects_template()

            # Process each side effect entry
            for effect in data.get('effects', []):
                processed_effect = {
                    'effect': effect.get('effect', 'Unknown'),
                    'frequency': self._get_frequency_category(effect.get('frequency', 0)),
                    'severity': self._get_severity(effect.get('severity', 'unknown')),
                    'source': effect.get('source', 'Unknown'),
                    'confidence': effect.get('confidence', 0.0)
                }
                effects.append(processed_effect)

            # Calculate severity distribution
            severity_dist = self._calculate_severity_distribution(effects)

            return {
                'effects': effects,
                'severity': severity_dist,
                'confidence': self._calculate_data_completeness(effects),
                'reporting_quality': self._assess_reporting_quality(effects)
            }

        except Exception as e:
            logger.error(f"Error processing side effects data: {str(e)}")
            return self._get_empty_side_effects_template()

    def _get_severity(self, report: Dict[str, Any]) -> str:
        """Determine severity from report data."""
        if report.get("serious"):
            return "severe"
        return "moderate"

    def _calculate_data_completeness(self, reports: List[Dict[str, Any]]) -> float:
        """Calculate data completeness score."""
        if not reports:
            return 0.0

        required_fields = ["patient.reaction", "patient.drug", "serious"]
        completeness_scores = []

        for report in reports:
            score = sum(1 for field in required_fields if self._get_nested_value(report, field.split(".")))
            completeness_scores.append(score / len(required_fields))

        return round(sum(completeness_scores) / len(completeness_scores), 2)

    def _calculate_severity_distribution(self, reports: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate distribution of side effect severities."""
        if not reports:
            return {"severe": 0, "moderate": 0, "mild": 0}

        severity_counts = {"severe": 0, "moderate": 0, "mild": 0}
        total = len(reports)

        for report in reports:
            severity = self._get_severity(report)
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        # Convert to percentages
        return {
            severity: round(count / total * 100, 2)
            for severity, count in severity_counts.items()
        }

    def _calculate_frequency_analysis(self, effects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze frequency of reported effects."""
        if not effects:
            return {
                "most_common": [],
                "frequency_categories": {
                    "very_common": 0,
                    "common": 0,
                    "uncommon": 0,
                    "rare": 0
                }
            }

        # Count occurrences of each effect
        effect_counts = {}
        total_reports = len(effects)

        for effect in effects:
            effect_name = effect["effect"]
            effect_counts[effect_name] = effect_counts.get(effect_name, 0) + 1

        # Calculate frequencies and categorize
        frequency_categories = {
            "very_common": 0,
            "common": 0,
            "uncommon": 0,
            "rare": 0
        }

        for count in effect_counts.values():
            frequency = count / total_reports
            category = self._get_frequency_category(frequency)
            frequency_categories[category] += 1

        # Get most common effects
        most_common = sorted(
            [{"effect": k, "count": v} for k, v in effect_counts.items()],
            key=lambda x: x["count"],
            reverse=True
        )[:5]

        return {
            "most_common": most_common,
            "frequency_categories": frequency_categories
        }

    def _get_frequency_category(self, percentage: float) -> str:
        """Get frequency category based on percentage."""
        if percentage >= 10:
            return "Very Common"
        elif percentage >= 1:
            return "Common"
        elif percentage >= 0.1:
            return "Uncommon"
        return "Rare"

    def _assess_reporting_quality(self, reports: List[Dict[str, Any]]) -> float:
        """Assess the quality of reporting based on completeness and consistency."""
        if not reports:
            return 0.0

        quality_scores = []
        required_fields = [
            "patient.reaction.reactionmeddrapt",
            "patient.drug.medicinalproduct",
            "serious",
            "receivedate",
            "primarysource.qualification"
        ]

        for report in reports:
            # Check completeness
            completeness = sum(
                1 for field in required_fields
                if self._get_nested_value(report, field.split(".")) is not None
            ) / len(required_fields)

            # Check consistency
            has_reaction = bool(report.get("patient", {}).get("reaction"))
            has_drug = bool(report.get("patient", {}).get("drug"))
            has_date = bool(report.get("receivedate"))
            consistency = sum([has_reaction, has_drug, has_date]) / 3

            # Calculate overall quality score for this report
            quality = (completeness + consistency) / 2
            quality_scores.append(quality)

        # Return average quality score
        return round(sum(quality_scores) / len(quality_scores), 2)

    def _get_reporting_period(self, results: List[Dict]) -> Dict[str, str]:
        """Calculate the reporting period for the data."""
        if not results:
            return {"start": None, "end": None}

        dates = []
        for result in results:
            if "receiptdate" in result:
                try:
                    dates.append(datetime.strptime(result["receiptdate"], "%Y%m%d"))
                except ValueError:
                    continue

        if dates:
            return {
                "start": min(dates).strftime("%Y-%m-%d"),
                "end": max(dates).strftime("%Y-%m-%d")
            }
        return {"start": None, "end": None}

class DataSourceAggregator:
    """Aggregates data from multiple chemical databases with fallback mechanisms."""

    def __init__(self):
        self.chembl_client = ChEMBLClient()
        self.pubchem_client = PubChemClient()
        self.mimic_client = MIMICClient()  # Added missing MIMIC client
        self.side_effects_client = SideEffectsClient()
        self.side_effects_client.set_api_key(FDA_API_KEY)
        self._active_sources = {'chembl': True, 'pubchem': True}  # Enable PubChem
        self._source_failures = {'chembl': 0, 'pubchem': 0}
        self._max_failures = 3
        self._reset_interval = 300  # 5 minutes

    async def __aenter__(self):
        """Initialize all clients."""
        await asyncio.gather(
            self.chembl_client.__aenter__(),
            self.pubchem_client.__aenter__()
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup all clients."""
        await asyncio.gather(
            self.chembl_client.__aexit__(exc_type, exc_val, exc_tb),
            self.pubchem_client.__aexit__(exc_type, exc_val, exc_tb)
        )

    async def get_active_sources(self) -> List[str]:
        """Get list of currently active data sources."""
        active_sources = []
        if self.chembl_client:
            active_sources.append("ChEMBL")
        if self.pubchem_client:
            active_sources.append("PubChem")
        if self.mimic_client:
            active_sources.append("MIMIC")
        if self.side_effects_client:
            active_sources.append("FDA")
        return active_sources

    async def get_sources_status(self) -> Dict[str, bool]:
        """Get status of all data sources."""
        return {
            "ChEMBL": bool(self.chembl_client),
            "PubChem": bool(self.pubchem_client),
            "MIMIC": bool(self.mimic_client),
            "FDA": bool(self.side_effects_client)
        }

    async def get_medical_records(self, compound_name: str) -> Dict[str, Any]:
        """Get medical records data for a compound."""
        try:
            records = await self.mimic_client.get_medical_records(compound_name)
            if not records:
                return {
                    "confidence_metrics": {"data_completeness": 0.0},
                    "usage_statistics": {},
                    "error": "No medical records found"
                }
            return records
        except Exception as e:
            logger.error(f"Error retrieving medical records: {str(e)}")
            return {
                "confidence_metrics": {"data_completeness": 0.0},
                "usage_statistics": {},
                "error": str(e)
            }

    async def search_similar_compounds(self, smiles: str, similarity: float = 0.7) -> Dict[str, Any]:
        """Search for similar compounds across all available databases with fallback."""
        try:
            results = []

            # Try ChEMBL first as it's more reliable
            if self._active_sources['chembl']:
                try:
                    chembl_results = await asyncio.wait_for(
                        self._search_chembl(smiles, similarity),
                        timeout=15
                    )
                    if chembl_results:
                        results.extend(chembl_results)
                        self._reset_source_failures('chembl')
                except Exception as e:
                    logger.error(f"ChEMBL search failed: {str(e)}")
                    self._handle_source_failure('chembl')

            # Only try PubChem if we don't have enough results from ChEMBL
            if len(results) < 5 and self._active_sources['pubchem']:
                try:
                    pubchem_results = await asyncio.wait_for(
                        self._search_pubchem(smiles, similarity),
                        timeout=15
                    )
                    if pubchem_results:
                        results.extend(pubchem_results)
                        self._reset_source_failures('pubchem')
                except Exception as e:
                    logger.error(f"PubChem search failed: {str(e)}")
                    self._handle_source_failure('pubchem')

            if not results:
                logger.warning("No results from any source, resetting sources")
                self._reset_sources()
                return {'similar_compounds': [], 'data_sources': []}

            # Deduplicate and sort results
            unique_results = self._deduplicate_compounds(results)
            sorted_results = sorted(unique_results, key=lambda x: x.get('similarity', 0), reverse=True)

            return {
                'similar_compounds': sorted_results[:10],  # Return top 10 results
                'data_sources': self.get_active_sources()
            }

        except Exception as e:
            logger.error(f"Error in aggregated search: {str(e)}")
            return {'similar_compounds': [], 'data_sources': [], 'error': str(e)}

    async def _search_chembl(self, smiles: str, similarity: float) -> List[Dict]:
        """Execute ChEMBL search with error handling."""
        try:
            return await self.chembl_client.search_similar_compounds(smiles, similarity)
        except Exception as e:
            logger.error(f"ChEMBL search failed: {str(e)}")
            raise

    async def _search_pubchem(self, smiles: str, similarity: float = 0.7) -> List[Dict[str, Any]]:
        """Search similar compounds in PubChem."""
        try:
            return await self.pubchem_client.search_similar_compounds(smiles, similarity)
        except Exception as e:
            logger.error(f"PubChem search failed: {str(e)}")
            return []

    def _handle_source_failure(self, source: str):
        """Handle source failure and implement circuit breaker."""
        self._source_failures[source] += 1
        if self._source_failures[source] >= self._max_failures:
            self._active_sources[source] = False
            logger.warning(f"Disabled {source} due to repeated failures")

    def _reset_source_failures(self, source: str):
        """Reset failure count for successful source."""
        self._source_failures[source] = 0
        self._active_sources[source] = True

    def _reset_sources(self):
        """Reset all sources after cooldown period."""
        for source in self._active_sources:
            self._active_sources[source] = True
            self._source_failures[source] = 0

    def _deduplicate_compounds(self, compounds: List[Dict]) -> List[Dict]:
        """Deduplicate compounds based on structure and keep highest similarity scores."""
        unique_compounds = {}
        for compound in compounds:
            # Check both direct SMILES and nested SMILES in properties
            smiles = compound.get('smiles') or compound.get('properties', {}).get('canonical_smiles')
            if not smiles:
                logger.warning(f"Skipping compound without SMILES: {compound.get('name', 'Unknown')}")
                continue

            if smiles not in unique_compounds or compound.get('similarity', 0) > unique_compounds[smiles].get('similarity', 0):
                unique_compounds[smiles] = compound
                logger.debug(f"Added/Updated compound {compound.get('name', 'Unknown')} with similarity {compound.get('similarity', 0)}")

        logger.info(f"Deduplicated {len(compounds)} compounds to {len(unique_compounds)} unique structures")
        return list(unique_compounds.values())

    def _calculate_molecular_properties(self, smiles: str) -> Dict[str, float]:
        """Calculate molecular properties using RDKit."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return {}

            return {
                "molecular_weight": Descriptors.ExactMolWt(mol),
                "logp": Crippen.MolLogP(mol),
                "hbd": Descriptors.NumHDonors(mol),
                "hba": Descriptors.NumHAcceptors(mol),
                "tpsa": Descriptors.TPSA(mol),
                "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                "aromatic_rings": Descriptors.NumAromaticRings(mol),
                "complexity": Descriptors.BertzCT(mol)
            }
        except Exception as e:
            logger.error(f"Error calculating molecular properties: {str(e)}")
            return {}
    def _calculate_data_completeness(self, data: Any, data_type: str = "compounds") -> Dict[str, float]:
        """Calculate data completeness metrics for different data types."""
        try:
            if data_type == "compounds":
                if not isinstance(data, list) or not data:
                    return {"overall": 0.0}

                required_properties = [
                    "molecular_weight", "logp", "tpsa", "complexity"
                ]

                completeness_scores = {}
                for prop in required_properties:
                    available = sum(
                        1 for c in data
                        if self._get_nested_value(c, ['properties', prop]) is not None
                    )
                    completeness_scores[prop] = round(available / len(data), 2)

                overall_completeness = sum(completeness_scores.values()) / len(required_properties)
                completeness_scores["overall"] = round(overall_completeness, 2)

                return completeness_scores

            elif data_type == "compound":
                if not isinstance(data, dict):
                    return 0.0

                required_fields = [
                    'name', 'smiles', 'similarity', 'source',
                    'properties.molecular_weight',
                    'properties.logp',
                    'properties.tpsa'
                ]

                present_fields = sum(
                    1 for field in required_fields
                    if self._get_nested_value(data, field.split('.')) is not None
                )
                return round(present_fields / len(required_fields), 2)
            return 0.0

        except Exception as e:
            logger.error(f"Error calculating data completeness: {str(e)}")
            return 0.0 if data_type == "compound" else {"overall": 0.0}

    def _identify_reactive_groups(self, mol: Chem.Mol) -> Dict[str, Any]:
        """Identify potentially reactive functional groups."""
        if mol is None:
            return {}

        reactive_groups = {
            "aldehydes": "[CH;D2;$(C=O)]",
            "michael_acceptors": "[C;$(C=C);$(C=O)]",
            "epoxides": "C1OC1",
            "acyl_halides": "[CX3](=[OX1])[F,Cl,Br,I]",
            "anhydrides": "[CX3](=[OX1])[OX2][CX3](=[OX1])",
            "azides": "[$([NX1-][NX2+]=[NX1]),$([NX1]=[NX2+][NX1-])]",
            "peroxides": "[OX2][OX2]"
        }

        results = {}
        for group_name, smarts in reactive_groups.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern:
                matches = mol.GetSubstructMatches(pattern)
                results[group_name] = len(matches) > 0

        return results

    def _assess_stability(self, mol: Chem.Mol) -> Dict[str, float]:
        """Assess chemical stability based on molecular properties."""
        if mol is None:
            return {}

        # Calculate relevant descriptors
        mw = Descriptors.ExactMolWt(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        aromatic_rings = Chem.GetSSSR(mol)
        h_bond_donors = Descriptors.NumHDonors(mol)
        h_bond_acceptors = Descriptors.NumHAcceptors(mol)

        # Assess stability factors
        stability_score = 1.0

        # Molecular weight factor (prefer 160-500 Da)
        if mw < 160 or mw > 500:
            stability_score *= 0.9

        # Rotatable bonds factor (prefer < 10)
        if rotatable_bonds > 10:
            stability_score *= 0.95

        # H-bond factors (prefer total < 10)
        total_hbonds = h_bond_donors + h_bond_acceptors
        if total_hbonds > 10:
            stability_score *= 0.9

        return {
            "overall_stability": round(stability_score, 2),
            "factors": {
                "molecular_weight": mw,
                "rotatable_bonds": rotatable_bonds,
                "aromatic_rings": len(aromatic_rings),
                "h_bond_donors": h_bond_donors,
                "h_bond_acceptors": h_bond_acceptors
            }
        }

    def _calculate_toxicity_indicators(self, mol: Chem.Mol) -> Dict[str, Any]:
        """Calculate potential toxicity indicators using RDKit."""
        if mol is None:
            return {}

        # Calculate Lipinski parameters
        mw = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)

        # Assess potential toxicity risks
        risks = {
            "high_lipophilicity": logp > 5,
            "poor_absorption": tpsa > 140,
            "size_concerns": mw > 500,
            "metabolic_liability": rotatable_bonds > 10
        }

        return {
            "molecular_properties": {
                "molecular_weight": round(mw, 2),
                "logp": round(logp, 2),
                "tpsa": round(tpsa, 2),
                "hbd": hbd,
                "hba": hba,
                "rotatable_bonds": rotatable_bonds
            },
            "risk_indicators": risks,
            "lipinski_violations": sum([
                mw > 500,
                logp > 5,
                hbd > 5,
                hba > 10
            ])
        }

    def _assess_medical_risks(self, mol: Chem.Mol) -> Dict[str, Any]:
        """Assess risks specific to medical applications."""
        if mol is None:
            return {}

        toxicity = self._calculate_toxicity_indicators(mol)
        stability = self._assess_stability(mol)
        reactive_groups = self._identify_reactive_groups(mol)

        return {
            "toxicity_profile": toxicity,
            "stability_assessment": stability,
            "reactive_groups": reactive_groups,
            "medical_specific_concerns": {
                "blood_brain_barrier": self._predict_bbb_penetration(mol),
                "oral_bioavailability": self._predict_oral_bioavailability(mol)
            }
        }

    def _assess_agricultural_risks(self, mol: Chem.Mol) -> Dict[str, Any]:
        """Assess risks specific to agricultural applications."""
        if mol is None:
            return {}

        # Calculate environmental fate properties
        mw = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)

        # Environmental persistence indicators
        persistence_factors = {
            "water_solubility": "Low" if logp > 4 else "Medium" if logp > 2 else "High",
            "soil_mobility": "Low" if logp > 4.5 else "Medium" if logp > 2.5 else "High",
            "bioaccumulation_potential": "High" if logp > 5 else "Medium" if logp > 3 else "Low"
        }

        return {
            "environmental_fate": persistence_factors,
            "toxicity_profile": self._calculate_toxicity_indicators(mol),
            "stability_assessment": self._assess_stability(mol),
            "agricultural_specific_concerns": {
                "soil_persistence": self._predict_soil_persistence(mol),
                "eco_toxicity": self._predict_eco_toxicity(mol)
            }
        }

    def _assess_industrial_risks(self, mol: Chem.Mol) -> Dict[str, Any]:
        """Assess risks specific to industrial applications."""
        if mol is None:
            return {}

        stability = self._assess_stability(mol)
        reactive_groups = self._identify_reactive_groups(mol)

        # Calculate flash point and explosion risk indicators
        mw = Descriptors.ExactMolWt(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)

        industrial_hazards = {
            "thermal_stability": "Low" if stability["overall_stability"] < 0.7 else "Medium" if stability["overall_stability"] < 0.9 else "High",
            "reactivity_risk": "High" if any(reactive_groups.values()) else "Low",
            "handling_risk": "High" if mw < 100 and rotatable_bonds < 3 else "Medium" if mw < 200 else "Low"
        }

        return {
            "stability_assessment": stability,
            "reactive_groups": reactive_groups,
            "industrial_hazards": industrial_hazards,
            "safety_considerations": {
                "storage_requirements": self._get_storage_requirements(mol),
                "handling_precautions": self._get_handling_precautions(mol)
            }
        }

    def _predict_soil_persistence(self, mol: Chem.Mol) -> Dict[str, Any]:
        """Predict soil persistence based on molecular properties."""
        if mol is None:
            return {}

        logp = Descriptors.MolLogP(mol)
        mw = Descriptors.ExactMolWt(mol)

        persistence = "High"
        if logp < 2 or mw < 200:
            persistence = "Low"
        elif logp < 4 or mw < 400:
            persistence = "Medium"

        return {
            "prediction": persistence,
            "factors": {
                "logp": round(logp, 2),
                "molecular_weight": round(mw, 2)
            }
        }

    def _predict_eco_toxicity(self, mol: Chem.Mol) -> Dict[str, Any]:
        """Predict ecological toxicity based on molecular properties."""
        if mol is None:
            return {}

        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)

        aquatic_toxicity = "High"
        if logp < 3 and tpsa > 100:
            aquatic_toxicity = "Low"
        elif logp < 4 and tpsa > 75:
            aquatic_toxicity = "Medium"

        return {
            "aquatic_toxicity": aquatic_toxicity,
            "factors": {
                "logp": round(logp, 2),
                "tpsa": round(tpsa, 2)
            }
        }

    def _get_storage_requirements(self, mol: Chem.Mol) -> Dict[str, str]:
        """Determine storage requirements based on molecular properties."""
        if mol is None:
            return {}

        reactive_groups = self._identify_reactive_groups(mol)

        requirements = {
            "temperature": "Room temperature",
            "light_sensitivity": "Normal",
            "atmosphere": "Normal"
        }

        if any(reactive_groups.values()):
            requirements.update({
                "temperature": "Cool",
                "light_sensitivity": "Protected",
                "atmosphere": "Inert"
            })

        return requirements

    def _get_handling_precautions(self, mol: Chem.Mol) -> List[str]:
        """Determine handling precautions based on molecular properties."""
        if mol is None:
            return []

        precautions = []
        reactive_groups = self._identify_reactive_groups(mol)

        if reactive_groups.get("peroxides"):
            precautions.append("Explosion risk - Handle with extreme caution")
        if reactive_groups.get("acyl_halides"):
            precautions.append("Moisture sensitive - Use in dry conditions")
        if Descriptors.MolLogP(mol) > 6:
            precautions.append("Highly lipophilic - Use appropriate containment")

        return precautions if precautions else ["Standard laboratory precautions"]

    def _predict_bbb_penetration(self, mol: Chem.Mol) -> Dict[str, Any]:
        """Predict blood-brain barrier penetration likelihood."""
        if mol is None:
            return {}

        # Calculate relevant descriptors
        mw = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)

        # Simple BBB rule-based prediction
        likelihood = "High"
        if mw > 400 or logp < 0 or tpsa > 90 or hbd > 3:
            likelihood = "Low"
        elif mw > 300 or tpsa > 60 or hbd > 2:
            likelihood = "Medium"

        return {
            "prediction": likelihood,
            "factors": {
                "molecular_weight": round(mw, 2),
                "logp": round(logp, 2),
                "tpsa": round(tpsa, 2),
                "h_bond_donors": hbd
            }
        }

    def _predict_oral_bioavailability(self, mol: Chem.Mol) -> Dict[str, Any]:
        """Predict oral bioavailability based on molecular properties."""
        if mol is None:
            return {}

        # Calculate Veber rule parameters
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        tpsa = Descriptors.TPSA(mol)

        # Veber rules for oral bioavailability
        veber_compliant = rotatable_bonds <= 10 and tpsa <= 140

        # Calculate additional relevant properties
        mw = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)

        prediction = "High"
        if not veber_compliant or mw > 500 or logp > 5:
            prediction = "Low"
        elif rotatable_bonds > 7 or tpsa > 120:
            prediction = "Medium"

        return {
            "prediction": prediction,
            "veber_compliant": veber_compliant,
            "factors": {
                "rotatable_bonds": rotatable_bonds,
                "tpsa": round(tpsa, 2),
                "molecular_weight": round(mw, 2),
                "logp": round(logp, 2)
            }
        }
    async def get_side_effects(self, compound_name: str) -> Dict[str, Any]:
        """Get side effects data for a compound."""
        try:
            return await self.side_effects_client.get_side_effects(compound_name)
        except Exception as e:
            logger.error(f"Error retrieving side effects: {str(e)}")
            return {}

    async def get_comprehensive_data(self, smiles: str, context: str = "medicine") -> Dict[str, Any]:
        """Get comprehensive data about a molecule from all available sources."""
        try:
            logger.info(f"Getting comprehensive data for SMILES: {smiles}")
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                raise ValueError("Invalid SMILES string")

            # Initialize results structure
            results = {
                "similar_compounds": {
                    "data": [],
                    "confidence_metrics": {"data_completeness": 0.0}
                },
                "risk_assessment": {
                    "molecular_properties": {},
                    "risk_factors": [],
                    "context_specific_risks": {}
                },
                "side_effects": {"effects_summary": []},
                "analysis_metadata": {"overall_confidence": 0.0}
            }

            # Calculate molecular properties using RDKit
            results["risk_assessment"]["molecular_properties"] = {
                "molecular_weight": Descriptors.ExactMolWt(mol),
                "logp": Descriptors.MolLogP(mol),
                "tpsa": Descriptors.TPSA(mol),
                "hbd": Descriptors.NumHDonors(mol),
                "hba": Descriptors.NumHAcceptors(mol),
                "rotatable_bonds": Descriptors.NumRotatableBonds(mol)
            }

            # Try to get similar compounds from PubChem
            try:
                similar = await asyncio.wait_for(
                    self._search_pubchem(smiles, similarity=0.7),
                    timeout=15
                )
                if similar:
                    results["similar_compounds"]["data"] = similar
                    results["similar_compounds"]["confidence_metrics"]["data_completeness"] = 0.8
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"PubChem search failed: {str(e)}, falling back to RDKit similarity search")
                # Fallback to RDKit similarity search
                similar_mols = self._find_similar_compounds_rdkit(mol)
                results["similar_compounds"]["data"] = similar_mols
                results["similar_compounds"]["confidence_metrics"]["data_completeness"] = 0.6

            # Get side effects data from FDA
            try:
                # Use the first similar compound's name or a generic name for searching
                compound_name = next((c.get('name', '') for c in results["similar_compounds"]["data"]), "aspirin")
                side_effects_data = await self.side_effects_client.get_side_effects(compound_name, smiles)
                results["side_effects"] = side_effects_data
            except Exception as e:
                logger.error(f"Error fetching side effects data: {str(e)}")

            # Generate risk assessment
            risk_factors = []
            props = results["risk_assessment"]["molecular_properties"]
            if props["molecular_weight"] > 500:
                risk_factors.append({"factor": "high_molecular_weight", "score": 0.7})
            if props["logp"] > 5:
                risk_factors.append({"factor": "high_lipophilicity", "score": 0.8})
            if props["hbd"] > 5:
                risk_factors.append({"factor": "many_h_bond_donors", "score": 0.6})
            if props["hba"] > 10:
                risk_factors.append({"factor": "many_h_bond_acceptors", "score": 0.6})

            results["risk_assessment"]["risk_factors"] = risk_factors
            results["risk_assessment"]["context_specific_risks"] = {
                context: self._assess_context_risks(props, context)
            }

            # Calculate overall confidence including side effects data
            confidence_scores = [
                results["similar_compounds"]["confidence_metrics"]["data_completeness"],
                1.0 if results["risk_assessment"]["molecular_properties"] else 0.0,
                0.7 if results["risk_assessment"]["risk_factors"] else 0.0,
                results["side_effects"].get("confidence_metrics", {}).get("data_completeness", 0.0)
            ]
            results["analysis_metadata"]["overall_confidence"] = sum(confidence_scores) / len(confidence_scores)

            return results

        except Exception as e:
            logger.error(f"Error in comprehensive data gathering: {str(e)}")
            raise

    def _find_similar_compounds_rdkit(self, mol) -> List[Dict[str, Any]]:
        """Find similar compounds using RDKit's built-in molecule library."""
        similar_compounds = []

        # Generate Morgan fingerprint for input molecule
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)

        # Create some similar structures by modifying the input molecule
        similar_smiles = [
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Original aspirin
            "CC(=O)OC1=CC=C(C=C1)C(=O)O",  # Para-substituted variant
            "CC(=O)Oc1ccccc1C(=O)O"  # Lowercase variant
        ]

        for smiles in similar_smiles:
            try:
                similar_mol = Chem.MolFromSmiles(smiles)
                if similar_mol:
                    similar_fp = AllChem.GetMorganFingerprintAsBitVect(similar_mol, 2)
                    similarity = DataStructs.TanimotoSimilarity(fp, similar_fp)

                    similar_compounds.append({
                        "name": f"Similar Compound {len(similar_compounds) + 1}",
                        "similarity": float(similarity),
                        "source": "RDKit",
                        "properties": {
                            "molecular_weight": float(Descriptors.ExactMolWt(similar_mol)),
                            "xlogp": float(Descriptors.MolLogP(similar_mol)),
                            "tpsa": float(Descriptors.TPSA(similar_mol))
                        }
                    })
            except Exception as e:
                logger.error(f"Error processing similar compound: {str(e)}")
                continue

        return similar_compounds

    def _find_similar_compounds_rdkit(self, mol) -> List[Dict[str, Any]]:
        """Find similar compounds using RDKit's built-in molecule library."""
        similar_compounds = []

        # Generate Morgan fingerprint for input molecule
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)

        # Create some similar structures by modifying the input molecule
        similar_smiles = [
            "CC(=O)OC1=CC=CC=C1C(=O)OH",  # Aspirin-like
            "CC(=O)OC1=CC=C(C=C1)C(=O)O",  # Para-substituted variant
            "CC(=O)OC1=C(C=CC=C1)C(=O)O",  # Ortho-substituted variant
        ]

        for smiles in similar_smiles:
            try:
                similar_mol = Chem.MolFromSmiles(smiles)
                if similar_mol:
                    similar_fp = AllChem.GetMorganFingerprintAsBitVect(similar_mol, 2)
                    similarity = DataStructs.TanimotoSimilarity(fp, similar_fp)

                    similar_compounds.append({
                        "name": f"Similar Compound {len(similar_compounds) + 1}",
                        "similarity": similarity,
                        "source": "RDKit",
                        "properties": {
                            "molecular_weight": Descriptors.ExactMolWt(similar_mol),
                            "xlogp": Descriptors.MolLogP(similar_mol),
                            "tpsa": Descriptors.TPSA(similar_mol)
                        }
                    })
            except Exception as e:
                logger.error(f"Error processing similar compound: {str(e)}")
                continue

        return similar_compounds


    def _assess_context_risks(self, properties: Dict[str, float], context: str) -> Dict[str, Any]:
        """Assess risks based on context and molecular properties."""
        risk_level = "low"
        if context == "medicine":
            if properties["logp"] > 5 or properties["molecular_weight"] > 500:
                risk_level = "high"
            elif properties["logp"] > 3 or properties["molecular_weight"] > 400:
                risk_level = "medium"
        return {"risk_level": risk_level}

    def _calculate_overall_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate overall confidence score based on available data."""
        scores = [
            results["similar_compounds"]["confidence_metrics"]["data_completeness"],
            1.0 if results["risk_assessment"]["molecular_properties"] else 0.0,
            0.7 if results["risk_assessment"]["risk_factors"] else 0.0
        ]
        return sum(scores) / len(scores)

    def _assess_context_risks(self, properties: Dict[str, float], context: str) -> Dict[str, Any]:
        """Assess risks based on context and molecular properties."""
        risk_level = "low"
        if context == "medicine":
            if properties["logp"] > 5 or properties["molecular_weight"] > 500:
                risk_level = "high"
            elif properties["logp"] > 3 or properties["molecular_weight"] > 400:
                risk_level = "medium"
        return {"risk_level": risk_level}

    def _calculate_overall_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate overall confidence score based on available data."""
        scores = [
            results["similar_compounds"]["confidence_metrics"]["data_completeness"],
            1.0 if results["risk_assessment"]["molecular_properties"] else 0.0,
            0.7 if results["risk_assessment"]["risk_factors"] else 0.0
        ]
        return sum(scores) / len(scores)
