"""
Multi-Agent Communication Module.

Provides message bus, handlers, and communication primitives for agent-to-agent
communication in PromptChain CLI.

FR-016 to FR-020: Agent Communication Bus
"""

from .handlers import cli_communication_handler, CommunicationHandler
from .message_bus import MessageBus, MessageType

__all__ = [
    "cli_communication_handler",
    "CommunicationHandler",
    "MessageBus",
    "MessageType",
]
