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

MODEL_NAME = "gpt-5-mini"

EMBEDDINGS_MODEL = "bge-m3:latest"
EMBEDDINGS_ENDPOINT = "http://localhost:11434/api/embed"
MODEL_ENDPOINT = (
    "https://ai-academy-foundry.openai.azure.com/openai/v1"
)
SYSTEM_PROMPT = ""
CHUNK_SIZE = 100
