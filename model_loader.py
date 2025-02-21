import torch
from huggingface_hub import snapshot_download
from gliner import GLiNER
from transformers import pipeline
from pii_config import GLINER_MODEL_PATH,ALPACA_MODEL_PATH, logger,DEVICE  # Import logger from config

def download_gliner():
    try:
        logger.info("Downloading GLiNER model...")
        snapshot_download(repo_id="knowledgator/gliner-multitask-large-v0.5", local_dir=GLINER_MODEL_PATH)

        logger.info("GLiNER model downloaded successfully.")
    except Exception as e:
        logger.error(f"Error downloading GLiNER model: {e}")


def download_model():
    try:
        logger.info("Downloading model...")
        snapshot_download(repo_id="declare-lab/flan-alpaca-gpt4-xl", local_dir=ALPACA_MODEL_PATH)
        logger.info("Model downloaded successfully.")
    except Exception as e:
        logger.error(f"Error downloading model: {e}")


def load_gliner():
    try:
        logger.info("Loading GLiNER model...")
        gliner_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GLiNER.from_pretrained(GLINER_MODEL_PATH).to(gliner_device)
        logger.info("GLiNER model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading GLiNER model: {e}")
        raise  # Re-raise the exception for debugging

def load_model():
    try:
        logger.info("Loading model...")
        model = pipeline("text2text-generation", model=ALPACA_MODEL_PATH, device=DEVICE)  # Explicit task
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
# Ensure GLiNER model is downloaded before usage
download_gliner()
download_model()
gliner_model = load_gliner()
alpaca_model = load_model()
