from dotenv import load_dotenv
import os

def get_openai_api_key():
    # Load variables from .env into environment
    load_dotenv()
    return os.getenv("OPENAI_API_KEY")

def safe_get(lst, index, default=None):
    return lst[index] if 0 <= index < len(lst) else default    