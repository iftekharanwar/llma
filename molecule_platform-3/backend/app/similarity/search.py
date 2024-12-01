from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, sascorer
from chembl_webresource_client.new_client import new_client
import pubchempy as pcp
from typing import List, Dict, Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

class MoleculeSimilaritySearch:
    def __init__(self):
        self.molecule = new_client.molecule
        self.activity = new_client.activity

    def generate_morgan_fingerprint(self, mol, radius=2):
        """Generate Morgan fingerprint for a molecule."""
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=2048)
        except Exception as e:
            logger.error(f"Error generating fingerprint: {str(e)}")
            return None

    def calculate_tanimoto_similarity(self, fp1, fp2) -> float:
        """Calculate Tanimoto similarity between two fingerprints."""
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    def search_pubchem(self, query_mol, min_similarity=0.7) -> List[Dict[str, Any]]:
        """Search PubChem for similar compounds."""
        query_fp = self.generate_morgan_fingerprint(query_mol)
        if not query_fp:
            return []

        similar_compounds = []
        try:
            # Search PubChem using SMILES
            smiles = Chem.MolToSmiles(query_mol)
            compounds = pcp.get_compounds(smiles, 'smiles')
            if not compounds:
                # Fallback to substructure search
                compounds = pcp.get_compounds(smiles, 'substructure')

            for compound in compounds:
                try:
                    mol = Chem.MolFromSmiles(compound.canonical_smiles)
                    if mol:
                        fp = self.generate_morgan_fingerprint(mol)
                        if fp:
                            similarity = self.calculate_tanimoto_similarity(query_fp, fp)
                            if similarity >= min_similarity:
                                # Generate 3D structure for similar molecule
                                mol_3d = Chem.AddHs(mol)
                                AllChem.EmbedMolecule(mol_3d)
                                AllChem.MMFFOptimizeMolecule(mol_3d)
                                mol_block_3d = Chem.MolToMolBlock(mol_3d)

                                similar_compounds.append({
                                    'cid': compound.cid,
                                    'smiles': compound.canonical_smiles,
                                    'similarity': similarity,
                                    'structure_3d': mol_block_3d,
                                    'iupac_name': compound.iupac_name,
                                    'molecular_weight': compound.molecular_weight,
                                    'source': 'PubChem'
                                })
                except Exception as e:
                    logger.warning(f"Error processing PubChem compound: {str(e)}")
                    continue
        except Exception as e:
            logger.error(f"Error searching PubChem: {str(e)}")

        return similar_compounds

    def search_chembl(self, query_mol, min_similarity=0.7) -> List[Dict[str, Any]]:
        """Search ChEMBL for similar compounds."""
        query_fp = self.generate_morgan_fingerprint(query_mol)
        if not query_fp:
            return []

        similar_compounds = []
        try:
            # Get SMILES for ChEMBL search
            smiles = Chem.MolToSmiles(query_mol)

            # Search ChEMBL using similarity
            similarity_results = self.molecule.filter(
                molecule_structures__canonical_smiles__flexmatch=smiles
            ).only(['molecule_chembl_id', 'molecule_structures'])

            for result in similarity_results:
                try:
                    mol = Chem.MolFromSmiles(
                        result['molecule_structures']['canonical_smiles']
                    )
                    if mol:
                        fp = self.generate_morgan_fingerprint(mol)
                        if fp:
                            similarity = self.calculate_tanimoto_similarity(query_fp, fp)
                            if similarity >= min_similarity:
                                # Get additional compound data
                                compound_data = self.molecule.get(result['molecule_chembl_id'])

                                # Generate 3D structure
                                mol_3d = Chem.AddHs(mol)
                                AllChem.EmbedMolecule(mol_3d)
                                AllChem.MMFFOptimizeMolecule(mol_3d)
                                mol_block_3d = Chem.MolToMolBlock(mol_3d)

                                # Get bioactivity data
                                activities = self.activity.filter(
                                    molecule_chembl_id=result['molecule_chembl_id']
                                ).only(['standard_type', 'standard_value', 'standard_units'])

                                similar_compounds.append({
                                    'chembl_id': result['molecule_chembl_id'],
                                    'smiles': result['molecule_structures']['canonical_smiles'],
                                    'similarity': similarity,
                                    'structure_3d': mol_block_3d,
                                    'iupac_name': compound_data.get('pref_name', ''),
                                    'molecular_weight': compound_data.get('molecule_properties', {}).get('full_mwt'),
                                    'bioactivity': [{
                                        'type': act['standard_type'],
                                        'value': act['standard_value'],
                                        'units': act['standard_units']
                                    } for act in activities],
                                    'source': 'ChEMBL'
                                })
                except Exception as e:
                    logger.warning(f"Error processing ChEMBL compound: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Error searching ChEMBL: {str(e)}")

        return similar_compounds

    def find_similar_molecules(self, structure: str, input_format: str = "smiles", min_similarity: float = 0.7) -> Dict[str, Any]:
        """Find similar molecules from multiple databases and analyze their properties."""
        try:
            # Parse input structure
            if input_format == "smiles":
                mol = Chem.MolFromSmiles(structure)
            elif input_format == "mol":
                mol = Chem.MolFromMolBlock(structure)
            elif input_format == "pdb":
                mol = Chem.MolFromPDBBlock(structure)
            else:
                raise ValueError(f"Unsupported input format: {input_format}")

            if mol is None:
                raise ValueError(f"Failed to parse structure in {input_format} format")

            # Search both databases in parallel
            similar_compounds = []
            similar_compounds.extend(self.search_pubchem(mol, min_similarity))
            similar_compounds.extend(self.search_chembl(mol, min_similarity))

            # Sort by similarity score
            similar_compounds.sort(key=lambda x: x['similarity'], reverse=True)

            # Calculate confidence metrics
            confidence = {
                'total_compounds': len(similar_compounds),
                'high_similarity_compounds': len([c for c in similar_compounds if c['similarity'] > 0.8]),
                'avg_similarity': np.mean([c['similarity'] for c in similar_compounds]) if similar_compounds else 0,
                'data_sources': list(set(c['source'] for c in similar_compounds))
            }

            return {
                'query_structure': structure,
                'similar_compounds': similar_compounds,
                'confidence_metrics': confidence
            }

        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return {
                'query_structure': structure,
                'similar_compounds': [],
                'confidence_metrics': {'error': str(e)}
            }

    def analyze_context(self, similar_compounds: List[Dict[str, Any]], context: str) -> Dict[str, Any]:
        """Analyze similar compounds in the context of their intended use."""
        try:
            # Context-specific analysis metrics
            context_metrics = {
                "medicine": {
                    "bioactivity_types": ["IC50", "EC50", "Ki", "Kd"],
                    "property_ranges": {
                        "molecular_weight": (160, 500),
                        "logp": (-0.4, 5.6)
                    }
                },
                "agriculture": {
                    "bioactivity_types": ["EC50", "LC50"],
                    "property_ranges": {
                        "molecular_weight": (200, 600),
                        "logp": (1.0, 4.5)
                    }
                }
            }

            metrics = context_metrics.get(context.lower(), context_metrics["medicine"])

            # Analyze compounds based on context
            context_analysis = {
                "compounds_in_range": 0,
                "bioactivity_matches": 0,
                "property_distribution": {
                    "molecular_weight": [],
                    "logp": []
                },
                "relevant_bioactivities": []
            }

            for compound in similar_compounds:
                # Check property ranges
                mw = compound.get('molecular_weight', 0)
                if metrics["property_ranges"]["molecular_weight"][0] <= mw <= metrics["property_ranges"]["molecular_weight"][1]:
                    context_analysis["compounds_in_range"] += 1

                # Collect property distributions
                context_analysis["property_distribution"]["molecular_weight"].append(mw)

                # Analyze bioactivities
                if 'bioactivity' in compound:
                    relevant_activities = [
                        activity for activity in compound['bioactivity']
                        if activity['type'] in metrics["bioactivity_types"]
                    ]
                    if relevant_activities:
                        context_analysis["bioactivity_matches"] += 1
                        context_analysis["relevant_bioactivities"].extend(relevant_activities)

            # Calculate context relevance score
            total_compounds = len(similar_compounds)
            if total_compounds > 0:
                context_analysis["context_relevance_score"] = round(
                    (context_analysis["compounds_in_range"] + context_analysis["bioactivity_matches"]) / (2 * total_compounds),
                    2
                )
            else:
                context_analysis["context_relevance_score"] = 0.0

            # Add statistical summaries
            if context_analysis["property_distribution"]["molecular_weight"]:
                context_analysis["property_statistics"] = {
                    "molecular_weight": {
                        "mean": np.mean(context_analysis["property_distribution"]["molecular_weight"]),
                        "std": np.std(context_analysis["property_distribution"]["molecular_weight"]),
                        "range": (min(context_analysis["property_distribution"]["molecular_weight"]),
                                max(context_analysis["property_distribution"]["molecular_weight"]))
                    }
                }

            return context_analysis

        except Exception as e:
            logger.error(f"Error in context analysis: {str(e)}")
            return {
                "error": str(e),
                "context_relevance_score": 0.0
            }

    def _calculate_confidence(self, similarity: float, record: Any) -> float:
        """Calculate confidence score based on similarity and record quality."""
        base_confidence = similarity * 100
        record_completeness = sum(1 for f in ['diagnosis', 'drug_name', 'outcome'] if hasattr(record, f) and getattr(record, f)) / 3
        return round(base_confidence * record_completeness, 1)

    def _analyze_molecular_properties(self, mol: Chem.Mol) -> Dict[str, Any]:
        """Analyze molecular properties using RDKit descriptors."""
        try:
            # Calculate basic molecular properties
            properties = {
                'molecular_weight': Descriptors.ExactMolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'aromatic_rings': Descriptors.NumAromaticRings(mol),
                'tpsa': Descriptors.TPSA(mol),
                'qed': Descriptors.qed(mol)
            }

            # Calculate additional drug-likeness properties
            properties.update({
                'lipinski_violations': sum([
                    1 if properties['molecular_weight'] > 500 else 0,
                    1 if properties['logp'] > 5 else 0,
                    1 if properties['hbd'] > 5 else 0,
                    1 if properties['hba'] > 10 else 0
                ]),
                'veber_violations': sum([
                    1 if properties['rotatable_bonds'] > 10 else 0,
                    1 if properties['tpsa'] > 140 else 0
                ])
            })

            # Add synthetic accessibility and natural product scores
            properties.update({
                'sa_score': self._calculate_sa_score(mol),
                'np_score': self._calculate_np_score(mol)
            })

            return properties

        except Exception as e:
            logger.error(f"Error analyzing molecular properties: {str(e)}")
            return {}
