"""
Application configuration.

This module contains all configurable settings used by the AI agent.

Future exercises may extend this file with:
- Model configuration
- API credentials
- Prompt templates
- Embedding settings
- Logging configuration
- Collage Professor Personality
"""
MODEL_NAME = "gemma4:e2b-it-qat"
MODEL_ENDPOINT = "http://localhost:11434/api/chat"
SYSTEM_PROMPT = "Esti un agent AI care ajuta utilizatorul sa raspunda la intrebari. Raspunde in limba romana."
