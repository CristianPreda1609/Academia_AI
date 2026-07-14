"""
Conversation memory management.

This module is responsible for storing and retrieving
messages exchanged between the user and the AI assistant.
"""


try:
    from .config import SYSTEM_PROMPT
except ImportError:
    from config import SYSTEM_PROMPT


class ConversationContext:
    def __init__(self):
        self.messages = [
            self.assemble_system_prompt()
        ]

    def assemble_system_prompt(self):
        # TODO: return a system message dict with the system prompt from config
        # Hint: Observe the message format used in agent.py
        # Hint: The system prompt should be a message dict with role "system"
        return {
            "role": "system",
            "content": SYSTEM_PROMPT
        }

    def add_message(self, message):
        # TODO: Implement message addition logic

        self.messages.append(message)

    def get_history(self):
        # TODO: return the full message history
        return self.messages
