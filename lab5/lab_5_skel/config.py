"""
Application configuration.

This module contains all configurable settings used by the AI agent.

Future exercises may extend this file with:
- Model configuration
- API credentials
- Prompt templates
- Embedding settings
- Logging configuration
"""

MODEL_NAME = "gemini-3.1-flash-lite"
API_KEY = ""
EMBEDDINGS_MODEL = "bge-m3:latest"
EMBEDDINGS_ENDPOINT = "http://localhost:11434/api/embed"
MODEL_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
)
SYSTEM_PROMPT = ""
CHUNK_SIZE = 100
TOP_N = 5
SIMILARITY_THRESHOLD = 0.5
