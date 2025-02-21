import os
import logging
import torch

GLINER_MODEL_PATH = r"./models/gliner"
ALPACA_MODEL_PATH = r"./models/alpaca"
PII_PATH = r"/pii_pro/data/PII.json"
SECTOR_DATA_PATH = r"/pii_pro/data/sectors.json"
CREDENTIAL_PII_PATH = r"pii_pro\data\credential_pii.json"
CREDENTIAL_PATH = r"pii_pro\data\credential.json"
GENERAL_PATH = r"pii_pro\data\general.json"
SENSITIVITY_PERSONAL_PATH = r"pii_pro\data\sensitivity_personal.json"
STANDARDS_PATH = r"pii_pro\data\standards.json"
VIOLATIONS_PATH = r"pii_pro\data\violations.json"


# Logging Setup
LOG_DIR = "app/logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# Device Configuration
DEVICE = 0 if torch.cuda.is_available() else -1