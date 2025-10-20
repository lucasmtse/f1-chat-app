import os
from dotenv import load_dotenv  # add this
load_dotenv()                   # and this
# Read API keys from environment variables (.env recommended)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "hide_your_key")
DEFAULT_MODEL = os.getenv("MISTRAL_MODEL", "mistral-medium")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
