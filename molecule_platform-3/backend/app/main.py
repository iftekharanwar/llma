import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import Optional, Dict, Any, List
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors
import io
import base64
import PyPDF2
import requests
import logging
from app.analysis.report_generator import ReportGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Initialize analysis components
report_generator = ReportGenerator()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # More permissive for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# AI API Configuration
AIML_API_KEY = os.getenv("AIML_API_KEY", "4c52a9817b884ef18597d098474086c1")
AIML_API_URL = os.getenv("AIML_API_URL", "https://api.aimlapi.com/api/v1/chat")

class MoleculeInput(BaseModel):
    structure: str
    input_format: str  # 'smiles', 'mol', 'pdb'
    context: str  # 'medicine', 'pesticide', 'sports', 'food'

    @validator('input_format')
    def validate_format(cls, v):
        if v not in ['smiles', 'mol', 'pdb']:
            raise ValueError('Invalid input format. Must be one of: smiles, mol, pdb')
        return v

    @validator('context')
    def validate_context(cls, v):
        if v not in ['medicine', 'pesticide', 'sports', 'food']:
            raise ValueError('Invalid context. Must be one of: medicine, pesticide, sports, food')
        return v

class SmilesInput(BaseModel):
    smiles: str

@app.post("/api/convert_smiles")
async def convert_smiles(input_data: SmilesInput):
    try:
        # Convert SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(input_data.smiles)
        if mol is None:
            raise HTTPException(status_code=400, detail="Invalid SMILES notation")

        # Generate 3D coordinates
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)

        # Convert to MOL format
        mol_block = Chem.MolToMolBlock(mol)

        return {"mol": mol_block}
    except Exception as e:
        logger.error(f"Error converting SMILES: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def parse_molecule(structure: str, input_format: str):
    """Parse molecular structure from various input formats."""
    mol = None
    try:
        if input_format == 'smiles':
            mol = Chem.MolFromSmiles(structure)
            if mol:
                # Generate 3D coordinates for SMILES input
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol)
                AllChem.MMFFOptimizeMolecule(mol)
        elif input_format == 'mol':
            mol = Chem.MolFromMolBlock(structure)
        elif input_format == 'pdb':
            mol = Chem.MolFromPDBBlock(structure)
            if mol:
                # Clean up PDB structure
                mol = Chem.RemoveHs(mol)
                AllChem.EmbedMolecule(mol)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing molecule: {str(e)}")

    if mol is None:
        raise HTTPException(status_code=400, detail=f"Failed to parse {input_format} structure")

    return mol

@app.post("/api/analyze")
async def analyze_molecule(molecule: MoleculeInput):
    logger.info(f"Received request with structure: {molecule.structure[:20]}... format: {molecule.input_format}")
    try:
        # Parse the molecule first
        logger.info("Attempting to parse molecule")
        mol = parse_molecule(molecule.structure, molecule.input_format)
        logger.info("Successfully parsed molecule")

        # Generate 2D depiction
        logger.info("Generating 2D depiction")
        img = Draw.MolToImage(mol)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_str = base64.b64encode(img_bytes.getvalue()).decode()
        logger.info("Generated 2D depiction")

        # Get 3D structure for visualization
        logger.info("Generating 3D structure")
        mol_3d = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_3d)
        AllChem.MMFFOptimizeMolecule(mol_3d)
        mol_block_3d = Chem.MolToMolBlock(mol_3d)
        logger.info("Generated 3D structure")

        # Generate comprehensive report using our new report generator
        logger.info("Generating comprehensive report")
        report = await report_generator.generate_comprehensive_report(
            molecule.structure,
            input_format=molecule.input_format
        )
        logger.info("Generated comprehensive report")

        # Return combined response with visualizations and report
        return {
            "image_2d": img_str,
            "structure_3d": mol_block_3d,
            "context": molecule.context,
            "input_format": molecule.input_format,
            "similarity_search_results": report["similarity_search_results"],
            "risk_assessment": report["risk_assessment"],
            "side_effects": report["side_effects"],
            "medical_records": report["medical_records"]
        }

    except Exception as e:
        logger.error(f"Error in analyze_molecule: {str(e)}")
        logger.exception("Full traceback:")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-pdf")
async def process_pdf(file: UploadFile = File(...)):
    try:
        # Read PDF content
        pdf_content = await file.read()
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Extract text from all pages
        text_content = ""
        for page in pdf_reader.pages:
            text_content += page.extract_text()

        # Basic structure extraction - look for common molecular formats
        import re
        # Look for SMILES patterns
        smiles_pattern = r'[A-Za-z0-9@+\-\[\]\(\)\\\/%=#$]+'
        potential_smiles = re.findall(smiles_pattern, text_content)

        # Look for molecular formulas
        formula_pattern = r'([A-Z][a-z]?\d*)+'
        potential_formulas = re.findall(formula_pattern, text_content)

        return {
            "text_content": text_content,
            "potential_structures": {
                "smiles": potential_smiles[:10],  # Limit to first 10 matches
                "formulas": potential_formulas[:10]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/api/chat")
async def chat_response(request: Request):
    """Handle chat interactions with molecular analysis context."""
    try:
        # Parse request data
        request_data = await request.json()
        question = request_data.get('question')
        molecule_data = request_data.get('molecule_data', {})

        if not question or not molecule_data:
            raise HTTPException(status_code=400, detail="Missing required fields: question and molecule_data")

        # Extract relevant data from molecule analysis
        properties = molecule_data.get("properties", {})
        risk_assessment = molecule_data.get("risk_assessment", {})
        similar_molecules = molecule_data.get("similar_molecules", [])
        context_analysis = molecule_data.get("context_analysis", {})

        # Format context for AI model
        context = f"""
        Analyzing molecule with the following data:
        Properties: {properties}
        Risk Assessment: {risk_assessment}
        Similar Natural Molecules: {similar_molecules}
        Context Analysis: {context_analysis}

        User Question: {question}
        """

        # Call AI API for response
        try:
            headers = {
                "Authorization": f"Bearer {AIML_API_KEY}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are SynthMol Assistant, an expert in molecular analysis and chemistry. Provide detailed, scientific responses about molecular properties, risks, and applications."
                    },
                    {
                        "role": "user",
                        "content": context
                    }
                ],
                "temperature": 0.7
            }

            logger.info("Sending request to AI API")
            response = requests.post(AIML_API_URL, headers=headers, json=payload, timeout=10)

            if response.status_code == 429:
                logger.warning("AI API rate limit exceeded")
                return {"response": _generate_fallback_response(question, properties, risk_assessment, similar_molecules, context_analysis)}

            response.raise_for_status()
            response_data = response.json()

            chatbot_response = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not chatbot_response:
                raise ValueError("Empty response from AI API")

            return {"response": chatbot_response}

        except requests.exceptions.RequestException as e:
            logger.error(f"AI API request failed: {str(e)}")
            return {"response": _generate_fallback_response(question, properties, risk_assessment, similar_molecules, context_analysis)}

    except Exception as e:
        logger.error(f"Error in chat response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

def _generate_fallback_response(question: str, properties: Dict, risk_assessment: Dict, similar_molecules: List, context_analysis: Dict) -> str:
    """Generate a fallback response when AI API is unavailable."""
    try:
        # Generate a context-aware response based on available data
        if "toxicity" in risk_assessment:
            risk_level = risk_assessment["toxicity"].get("level", "unknown")
            risk_details = risk_assessment["toxicity"].get("details", [])
        else:
            risk_level = "unknown"
            risk_details = []

        if question.lower().find("risk") >= 0 or question.lower().find("safety") >= 0:
            return f"Based on our analysis, this molecule shows {risk_level} risk level. " + \
                   (f"Key considerations include: {', '.join(risk_details[:3])}. " if risk_details else "")

        if question.lower().find("similar") >= 0:
            if similar_molecules:
                return f"The molecule shows similarity to {len(similar_molecules)} known compounds. " + \
                       f"The most similar natural molecule is {similar_molecules[0].get('name', 'unknown')}."
            return "No similar molecules were found in our database."

        if question.lower().find("property") >= 0 or question.lower().find("properties") >= 0:
            property_list = [f"{k}: {v}" for k, v in properties.items()][:3]
            if property_list:
                return f"Key properties of this molecule include: {', '.join(property_list)}."
            return "No specific property information is available for this molecule."

        # Default response
        return "I understand you're asking about this molecule. While I'm currently experiencing connection issues, " + \
               "I can tell you that our analysis has identified its basic properties and safety profile. " + \
               "Please try asking a more specific question about its risks, properties, or similar compounds."
    except Exception as e:
        logger.error(f"Error generating fallback response: {str(e)}")
        return "I apologize, but I'm currently unable to provide a detailed response. Please try again later."

@app.get("/api/medical-records/{molecule_id}")
async def get_medical_records(molecule_id: str):
    """Fetch medical records related to a molecule based on similarity search."""
    try:
        # Query ChEMBL for medical/clinical data
        chembl_url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{molecule_id}/clinical_data"
        response = requests.get(chembl_url)

        if not response.ok:
            return {"records": []}

        clinical_data = response.json()

        # Transform ChEMBL clinical data into our format
        records = []
        for data in clinical_data.get("clinical_data", []):
            record = {
                "id": data.get("document_id"),
                "age": data.get("age_group", "Unknown"),
                "gender": data.get("subject_sex", "Not specified"),
                "condition": data.get("condition_name", "Unknown"),
                "response": _categorize_response(data.get("max_phase_for_ind", 0))
            }
            records.append(record)

        return {"records": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching medical records: {str(e)}")

def _categorize_response(phase: int) -> str:
    """Categorize clinical trial phase into response type."""
    if phase >= 4:
        return "Positive"
    elif phase >= 2:
        return "Mixed"
    elif phase >= 1:
        return "Neutral"
    else:
        return "Negative"

@app.get("/")
async def root():
    return {"message": "Molecular Analysis Platform API"}
