from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'efficientnet_b0_best.pth'
CLASS_MAPPING_PATH = PROJECT_ROOT / 'config' / 'class_mapping.json'
DATABASE_PATH = PROJECT_ROOT / 'database' / 'analysis_history.db'
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / 'knowledge_base'
REPORTS_DIR = PROJECT_ROOT / 'reports'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
SAMPLE_IMAGES_DIR = PROJECT_ROOT / 'sample_images'

DEFAULT_DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'

EXPECTED_NUM_CLASSES = 38

def ensure_model_artifacts() -> tuple[bool, str]:
    if not MODEL_PATH.exists():
        return False, (
            'Missing trained checkpoint: "models/efficientnet_b0_best.pth". '
            'Place your actual EfficientNet-B0 state_dict checkpoint in this location '
            'before running disease prediction.'
        )
    if not CLASS_MAPPING_PATH.exists():
        return False, (
            'Missing class mapping file: "config/class_mapping.json". '
            'This file must contain the trained model label-to-disease mapping '
            'for the 38 output classes.'
        )
    return True, 'Model artifacts are available.'
