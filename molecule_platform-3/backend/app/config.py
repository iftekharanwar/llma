import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database Configuration
MIMIC_DB_CONFIG = {
    'dbname': os.getenv('MIMIC_DB_NAME', 'mimic'),
    'user': os.getenv('MIMIC_DB_USER', 'postgres'),
    'password': os.getenv('MIMIC_DB_PASSWORD', ''),
    'host': os.getenv('MIMIC_DB_HOST', 'localhost'),
    'port': os.getenv('MIMIC_DB_PORT', '5432')
}

# API Configuration
CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/v2"
PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
FDA_API_KEY = os.getenv('FDA_API_KEY')

# Data Source Settings
SIMILARITY_THRESHOLD = 0.7
MAX_SIMILAR_COMPOUNDS = 10
CACHE_TIMEOUT = 3600  # 1 hour
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3

# API Rate Limiting
RATE_LIMIT = {
    'pubchem': {'requests': 5, 'per_seconds': 1},
    'chembl': {'requests': 3, 'per_seconds': 1},
    'fda': {'requests': 3, 'per_seconds': 1}
}

# Logging Configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': True
        }
    }
}

# Report Generation Settings
REPORT_SETTINGS = {
    'max_medical_records': 1000,
    'min_confidence_score': 0.6,
    'max_side_effects': 50,
    'context_weights': {
        'medicine': {
            'toxicity': 0.3,
            'efficacy': 0.3,
            'side_effects': 0.2,
            'interactions': 0.2
        },
        'agriculture': {
            'environmental_impact': 0.3,
            'toxicity': 0.3,
            'efficacy': 0.2,
            'persistence': 0.2
        },
        'sports': {
            'safety': 0.4,
            'performance': 0.3,
            'legality': 0.3
        }
    },
    'similarity_thresholds': {
        'high': 0.9,
        'medium': 0.7,
        'low': 0.5
    },
    'risk_assessment': {
        'toxicity_threshold': 0.7,
        'reactivity_threshold': 0.6,
        'environmental_impact_threshold': 0.5
    },
    'side_effects': {
        'severity_weights': {
            'severe': 1.0,
            'moderate': 0.6,
            'mild': 0.3
        },
        'frequency_weights': {
            'very_common': 1.0,
            'common': 0.7,
            'uncommon': 0.4,
            'rare': 0.2
        }
    }
}

# Error Messages
ERROR_MESSAGES = {
    'api_timeout': 'API request timed out. Please try again.',
    'rate_limit': 'Rate limit exceeded. Please wait before trying again.',
    'invalid_smiles': 'Invalid SMILES string provided.',
    'no_data_found': 'No data found for the given molecule.',
    'insufficient_confidence': 'Insufficient confidence in analysis results.'
}
