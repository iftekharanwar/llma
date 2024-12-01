import os
import logging
import urllib.parse
from typing import List, Dict, Any
import requests
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs
from ..data_sources.api_clients import DataSourceAggregator

logger = logging.getLogger(__name__)

class SimilaritySearch:
    def __init__(self):
        self.data_aggregator = DataSourceAggregator()

        # Verify RDKit is properly initialized
        test_mol = Chem.MolFromSmiles("CC")
        if not test_mol:
            logger.error("RDKit initialization failed")
            raise RuntimeError("Failed to initialize RDKit")

    def check_pubchem_status(self) -> str:
        """Check PubChem API status."""
        try:
            response = requests.get(f"{self.data_aggregator.pubchem_client.BASE_URL}/compound/random/JSON")
            return "active" if response.status_code == 200 else "error"
        except Exception:
            return "error"

    def check_chembl_status(self) -> str:
        """Check ChEMBL API status."""
        try:
            response = requests.get(f"{self.data_aggregator.chembl_client.BASE_URL}/status")
            return "active" if response.status_code == 200 else "error"
        except Exception:
            return "error"

    async def search_similar_compounds(self, structure: str, input_format: str = "smiles", min_similarity: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar compounds using both PubChem and ChEMBL APIs."""
        try:
            # Convert input to RDKit molecule
            if input_format.lower() == "smiles":
                query_mol = Chem.MolFromSmiles(structure)
            elif input_format.lower() == "mol":
                query_mol = Chem.MolFromMolBlock(structure)
            elif input_format.lower() == "pdb":
                query_mol = Chem.MolFromPDBBlock(structure)
            else:
                raise ValueError(f"Unsupported input format: {input_format}")

            if query_mol is None:
                raise ValueError("Failed to parse input structure")

            # Use the DataSourceAggregator to search for similar compounds
            similar_compounds = await self.data_aggregator.search_similar_compounds(
                structure if input_format.lower() == "smiles" else Chem.MolToSmiles(query_mol),
                min_similarity
            )

            return similar_compounds

        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise

    async def _search_pubchem(self, structure: str, min_similarity: float) -> List[Dict[str, Any]]:
        """Search PubChem for similar compounds."""
        try:
            # Encode SMILES for URL
            encoded_smiles = urllib.parse.quote(structure)
            similar_compounds = []

            # First, get the CID for the input structure
            response = requests.get(
                f"{self.pubchem_base_url}/compound/smiles/{encoded_smiles}/cids/JSON"
            )
            if response.status_code != 200:
                logger.warning(f"PubChem CID lookup failed: {response.status_code}")
                return []

            data = response.json()
            if 'IdentifierList' not in data:
                return []

            query_cid = data['IdentifierList']['CID'][0]

            # Search for similar compounds using 2D similarity
            similarity_url = (
                f"{self.pubchem_base_url}/compound/fastsimilarity_2d/cid/{query_cid}"
                f"/property/IUPACName,MolecularFormula,InChI,CanonicalSMILES/JSON"
            )

            response = requests.get(similarity_url)
            if response.status_code != 200:
                logger.warning(f"PubChem similarity search failed: {response.status_code}")
                return []

            data = response.json()
            if 'PropertyTable' not in data:
                return []

            for compound in data['PropertyTable']['Properties']:
                # Calculate actual similarity using RDKit
                mol = Chem.MolFromSmiles(compound.get('CanonicalSMILES', ''))
                if not mol:
                    continue

                similarity = self._calculate_similarity(
                    Chem.MolFromSmiles(structure),
                    mol
                )

                if similarity >= min_similarity:
                    properties = self._extract_molecular_properties(mol)
                    similar_compounds.append({
                        'name': compound.get('IUPACName', 'Unknown'),
                        'smiles': compound.get('CanonicalSMILES', ''),
                        'inchi': compound.get('InChI', ''),
                        'formula': compound.get('MolecularFormula', ''),
                        'similarity': similarity,
                        'source': 'PubChem',
                        'cid': compound.get('CID', ''),
                        'properties': properties,
                        'data_quality': self._assess_data_quality(compound)
                    })

            return similar_compounds

        except Exception as e:
            logger.error(f"Error in PubChem search: {str(e)}")
            return []

    async def _search_chembl(self, structure: str, min_similarity: float) -> List[Dict[str, Any]]:
        """Search ChEMBL for similar compounds."""
        try:
            similar_compounds = []

            # First get the ChEMBL ID for the input structure
            encoded_smiles = urllib.parse.quote(structure)
            response = requests.get(
                f"{self.chembl_base_url}/molecule/smiles/{encoded_smiles}"
            )

            if response.status_code != 200:
                logger.warning(f"ChEMBL structure lookup failed: {response.status_code}")
                return []

            molecule_data = response.json()
            if not molecule_data:
                return []

            chembl_id = molecule_data.get('molecule_chembl_id')

            # Search for similar molecules using substructure search
            similarity_url = (
                f"{self.chembl_base_url}/similarity/{encoded_smiles}/70"
            )

            response = requests.get(similarity_url)
            if response.status_code != 200:
                logger.warning(f"ChEMBL similarity search failed: {response.status_code}")
                return []

            molecules = response.json().get('molecules', [])

            for mol_data in molecules:
                mol = Chem.MolFromSmiles(mol_data.get('molecule_structures', {}).get('canonical_smiles', ''))
                if not mol:
                    continue

                similarity = self._calculate_similarity(
                    Chem.MolFromSmiles(structure),
                    mol
                )

                if similarity >= min_similarity:
                    # Get detailed properties from ChEMBL
                    properties = self._extract_chembl_properties(mol_data)
                    properties.update(self._calculate_rdkit_properties(mol))

                    similar_compounds.append({
                        'name': mol_data.get('pref_name', 'Unknown'),
                        'smiles': mol_data.get('molecule_structures', {}).get('canonical_smiles', ''),
                        'similarity': similarity,
                        'source': 'ChEMBL',
                        'chembl_id': mol_data.get('molecule_chembl_id', ''),
                        'properties': properties,
                        'data_quality': self._assess_data_quality(mol_data)
                    })

            return similar_compounds

        except Exception as e:
            logger.error(f"Error in ChEMBL search: {str(e)}")
            return []

    def _calculate_similarity(self, mol1, mol2) -> float:
        """Calculate Morgan fingerprint similarity between two molecules."""
        if not mol1 or not mol2:
            return 0.0

        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, 1024)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 1024)
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    def _extract_molecular_properties(self, mol) -> Dict[str, Any]:
        """Extract molecular properties using RDKit."""
        try:
            return {
                'molecular_weight': Descriptors.ExactMolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'tpsa': Descriptors.TPSA(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'rings': Descriptors.RingCount(mol),
                'aromatic_rings': Descriptors.NumAromaticRings(mol)
            }
        except Exception as e:
            logger.error(f"Error extracting molecular properties: {str(e)}")
            return {}

    def _extract_chembl_properties(self, mol_data: Dict) -> Dict[str, Any]:
        """Extract properties from ChEMBL molecule data."""
        try:
            properties = {}

            # Extract molecule properties
            if 'molecule_properties' in mol_data:
                mp = mol_data['molecule_properties']
                properties.update({
                    'alogp': mp.get('alogp', None),
                    'psa': mp.get('psa', None),
                    'hba': mp.get('hba', None),
                    'hbd': mp.get('hbd', None),
                    'num_ro5_violations': mp.get('num_ro5_violations', None),
                    'rtb': mp.get('rtb', None),
                    'full_mwt': mp.get('full_mwt', None),
                    'aromatic_rings': mp.get('aromatic_rings', None)
                })

            # Add bioactivity data if available
            if 'molecule_chembl_id' in mol_data:
                bioactivity_url = f"{self.chembl_base_url}/mechanism/{mol_data['molecule_chembl_id']}"
                response = requests.get(bioactivity_url)
                if response.status_code == 200:
                    mechanisms = response.json().get('mechanisms', [])
                    if mechanisms:
                        properties['known_mechanisms'] = [
                            {
                                'action_type': m.get('action_type', ''),
                                'mechanism_of_action': m.get('mechanism_of_action', ''),
                                'target_name': m.get('target_name', '')
                            }
                            for m in mechanisms
                        ]

            return properties
        except Exception as e:
            logger.error(f"Error extracting ChEMBL properties: {str(e)}")
            return {}

    def _calculate_rdkit_properties(self, mol) -> Dict[str, float]:
        """Calculate additional RDKit properties."""
        try:
            return {
                'qed': Descriptors.qed(mol),  # Drug-likeness score
                'sas': Descriptors.sas(mol),  # Synthetic accessibility
                'bertz': Descriptors.BertzCT(mol),  # Complexity
                'charge': Descriptors.MinAbsPartialCharge(mol)  # Charge-related
            }
        except Exception as e:
            logger.error(f"Error calculating RDKit properties: {str(e)}")
            return {}

    def _assess_data_quality(self, compound_data: Dict) -> Dict[str, Any]:
        """Assess the quality and completeness of compound data."""
        try:
            required_fields = ['name', 'smiles', 'properties']
            optional_fields = ['inchi', 'formula', 'mechanisms']

            # Calculate completeness score
            required_complete = sum(1 for field in required_fields if field in compound_data) / len(required_fields)
            optional_complete = sum(1 for field in optional_fields if field in compound_data) / len(optional_fields)

            return {
                'completeness_score': round(0.7 * required_complete + 0.3 * optional_complete, 2),
                'has_3d_structure': 'structure_3d' in compound_data,
                'has_experimental_data': bool(compound_data.get('properties', {}).get('known_mechanisms', [])),
                'property_coverage': len(compound_data.get('properties', {})) / 8  # Normalized by expected properties
            }
        except Exception as e:
            logger.error(f"Error assessing data quality: {str(e)}")
            return {}

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate compounds based on SMILES."""
        seen_smiles = set()
        unique_results = []

        for result in results:
            if result["smiles"] not in seen_smiles:
                seen_smiles.add(result["smiles"])
                unique_results.append(result)

        # Sort by similarity score
        return sorted(unique_results, key=lambda x: x["similarity"], reverse=True)

    def check_pubchem_status(self) -> str:
        """Check PubChem API status."""
        try:
            response = requests.get(f"{self.pubchem_base_url}/ping")
            return "active" if response.status_code == 200 else "inactive"
        except:
            return "error"

    def check_chembl_status(self) -> str:
        """Check ChEMBL API status."""
        try:
            response = requests.get(f"{self.chembl_base_url}/status")
            return "active" if response.status_code == 200 else "inactive"
        except:
            return "error"
