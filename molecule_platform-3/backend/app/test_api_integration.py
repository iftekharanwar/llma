```python
import asyncio
import logging
from data_sources.api_clients import DataSourceAggregator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_similarity_search():
    """Test similarity search functionality with aspirin as an example."""
    try:
        # Aspirin SMILES structure
        aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"

        aggregator = DataSourceAggregator()
        results = await aggregator.search_similar_compounds(aspirin_smiles, min_similarity=0.7)

        logger.info(f"Found {len(results)} similar compounds")

        # Log first few results
        for i, compound in enumerate(results[:5]):
            logger.info(f"\nCompound {i+1}:")
            logger.info(f"Name: {compound.get('name')}")
            logger.info(f"Similarity: {compound.get('similarity')}")
            logger.info(f"Source: {compound.get('source')}")
            logger.info(f"Properties: {compound.get('properties')}")

    except Exception as e:
        logger.error(f"Error in similarity search test: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_similarity_search())
```
