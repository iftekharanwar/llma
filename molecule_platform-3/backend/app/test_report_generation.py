import asyncio
from typing import Dict, Any
import logging
from datetime import datetime
from app.analysis.report_generator import ReportGenerator
from app.analysis.medical_records import MedicalRecordsAnalyzer
from app.analysis.side_effects import SideEffectsAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_report_generation():
    """Test comprehensive report generation with real data integration."""
    try:
        # Test molecule: Aspirin SMILES structure
        aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"

        # Initialize analyzers with debug logging
        logging.getLogger('analysis.similarity_search').setLevel(logging.DEBUG)
        logging.getLogger('analysis.medical_records').setLevel(logging.DEBUG)
        logging.getLogger('analysis.side_effects').setLevel(logging.DEBUG)
        logging.getLogger('analysis.risk_assessment').setLevel(logging.DEBUG)

        report_gen = ReportGenerator()
        medical_analyzer = MedicalRecordsAnalyzer()
        side_effects_analyzer = SideEffectsAnalyzer()

        # Generate comprehensive report
        logger.info("\n=== Generating Comprehensive Report for Aspirin ===")
        logger.info(f"Input SMILES: {aspirin_smiles}")

        report = await report_gen.generate_comprehensive_report(
            structure=aspirin_smiles,
            context="medicine",
            input_format="smiles"
        )

        # Validate similarity search results have real data
        logger.info("\n=== Validating Similarity Search Results ===")
        similar_compounds = report["similarity_search_results"]["similar_compounds"]
        if not similar_compounds:
            raise ValueError("No similar compounds found - potential API connection issue")

        # Verify we have real compound data
        for compound in similar_compounds[:3]:
            if not all(k in compound for k in ['name', 'similarity', 'source', 'known_properties']):
                raise ValueError(f"Missing required compound data fields in {compound['name']}")
            if not isinstance(compound['similarity'], float) or not (0 <= compound['similarity'] <= 1):
                raise ValueError(f"Invalid similarity score for {compound['name']}")
            logger.info(f"Validated compound: {compound['name']} (similarity: {compound['similarity']:.2f})")

        # Validate risk assessment contains real analysis
        logger.info("\n=== Validating Risk Assessment ===")
        risk_data = report["risk_assessment"]
        required_risk_fields = ['molecular_properties', 'toxicity_assessment', 'context_specific_risks']
        if not all(field in risk_data for field in required_risk_fields):
            raise ValueError("Missing required risk assessment fields")

        # Verify molecular properties contain actual calculations
        mol_props = risk_data["molecular_properties"]
        if not mol_props or all(v == 0 for v in mol_props.values()):
            raise ValueError("Molecular properties appear to be placeholder data")
        logger.info("Validated molecular properties calculations")

        # Validate medical records contain real data
        logger.info("\n=== Validating Medical Records Analysis ===")
        medical_data = report["medical_records_analysis"]
        if not medical_data['usage_statistics'].get('total_prescriptions', 0) > 0:
            raise ValueError("Medical records appear to lack real prescription data")
        logger.info(f"Validated {medical_data['usage_statistics']['total_prescriptions']} prescription records")

        # Validate side effects analysis contains real data
        logger.info("\n=== Validating Side Effects Analysis ===")
        side_effects = report["side_effects_analysis"]
        if not side_effects['effects_summary'].get('total_effects', 0) > 0:
            raise ValueError("Side effects analysis appears to lack real data")
        logger.info(f"Validated {side_effects['effects_summary']['total_effects']} side effects")

        # Validate confidence metrics
        logger.info("\n=== Validating Data Quality Metrics ===")
        confidence = report["metadata"]["confidence_metrics"]
        if not (0 <= confidence['overall_confidence'] <= 100):
            raise ValueError("Invalid confidence score range")
        logger.info(f"Overall confidence: {confidence['overall_confidence']}%")

        # Validate API status
        logger.info("\n=== Validating API Status ===")
        api_status = report["metadata"]["api_status"]
        if not all(status == "active" for status in api_status.values()):
            logger.warning("Some APIs may be inactive: " + str(api_status))

        return report

    except Exception as e:
        logger.error(f"Error in report generation test: {str(e)}")
        logger.error("Full error details:", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(test_report_generation())
