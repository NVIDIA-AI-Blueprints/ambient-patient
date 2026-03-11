# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from langgraph.graph.message import AnyMessage, add_messages
from langchain.messages import HumanMessage, AIMessage
from langchain_core.messages.tool import ToolMessage
import re
import logging
import os

log_level_str = os.getenv('LOG_LEVEL', 'INFO').upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)


def add_messages_with_reset(left: list[AnyMessage], right: list[AnyMessage] or AnyMessage) -> list[AnyMessage]:
    """
    Merges two lists of messages, with an option to clear all if a new message contains "start over".

    Args:
        left: The base list of messages.
        right: The list of messages (or single message) to merge
            into the base list.

    Returns:
        A new list of messages. If a message in `right` contains "start over",
        all previous messages are cleared and only this message is returned.
    """
    if not isinstance(right, list):
        right = [right]
    
    # Check for "start over" in the content of any new message
    
    right_str = []
    if isinstance(right[0], HumanMessage) or isinstance(right[0], AIMessage):
        right_str = [m.content for m in right if isinstance(m, HumanMessage) ]
    elif isinstance(right[0], dict):
        right_str = [m['content'] for m in right if m['role'] == 'user']
    elif isinstance(right[0], ToolMessage):
        pass
    else:
        logger.warning(f"Right messages which is a list of type {type(right[0])} in add_messages_with_reset: {right}")
        logger.warning(f"Not utilizing add_messages_with_reset and falling back to add_messages")
        return add_messages(left, right)

    # Check for "start over" in the content of any new message
    
    pattern = r"\b(restart|start over|a new session)\b"
    for m_str in right_str:
        
        needs_reset = re.search(pattern, m_str, re.IGNORECASE)

        if needs_reset:
            logger.warning(f"Resetting messages to only one 'Hi' message in add_messages_with_reset, from message content: {m_str}")
            
            return [{'role': 'user', 'content': 'Hi'}]

    return add_messages(left, right)
