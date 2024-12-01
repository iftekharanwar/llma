import pytest
import asyncio
from app.analysis.comprehensive_report import ComprehensiveReportGenerator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_comprehensive_report_generation():
    """Test comprehensive report generation with real molecular data."""
    try:
        # Initialize report generator
        generator = ComprehensiveReportGenerator()

        # Test molecules with known properties
        test_cases = [
            {
                "name": "Aspirin",
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "context": "medicine"
            },
            {
                "name": "Caffeine",
                "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                "context": "food"
            }
        ]

        for test_case in test_cases:
            logger.info(f"Testing report generation for {test_case['name']}")

            # Generate report
            report = await generator.generate_report(
                structure=test_case["smiles"],
                context=test_case["context"]
            )

            # Verify report structure and content
            assert "similarity_search_results" in report
            assert "risk_assessment" in report
            assert "analysis_metadata" in report

            # Verify similarity search results
            similar_compounds = report["similarity_search_results"]["similar_compounds"]
            assert len(similar_compounds) > 0, f"No similar compounds found for {test_case['name']}"

            # Verify risk assessment
            risk_data = report["risk_assessment"]
            assert "toxicity_assessment" in risk_data
            assert "context_specific_risks" in risk_data

            # Verify data completeness and confidence scores
            metadata = report["analysis_metadata"]
            assert metadata["data_completeness"] > 0
            assert metadata["confidence_score"] > 0

            # Log results
            logger.info(f"Report generated successfully for {test_case['name']}")
            logger.info(f"Found {len(similar_compounds)} similar compounds")
            logger.info(f"Data completeness: {metadata['data_completeness']}%")
            logger.info(f"Confidence score: {metadata['confidence_score']}%")

    except Exception as e:
        logger.error(f"Error in comprehensive report test: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(test_comprehensive_report_generation())
