"""
actions/whatsapp.py - Send WhatsApp messages via pywhatkit (WhatsApp Web).

Requirements:
  - Chrome browser open with WhatsApp Web logged in
  - pywhatkit installed

Note: pywhatkit uses browser automation (PyAutoGUI). It's not 100% reliable
but works for personal use. Do NOT commit contacts or phone numbers to Git.
"""

import logging
import re
import asyncio
import os
from config import WHATSAPP_DEFAULT_WAIT

logger = logging.getLogger("Charlie.whatsapp")

# Common contacts - store here or load from a local contacts.json (gitignored)
# Format: {"name": "+91XXXXXXXXXX"}
CONTACTS: dict[str, str] = {}

try:
    import json, os
    contacts_path = os.path.join(os.path.dirname(__file__), "..", "data", "contacts.json")
    if os.path.exists(contacts_path):
        with open(contacts_path) as f:
            CONTACTS = json.load(f)
        logger.info("Loaded %d contacts.", len(CONTACTS))
except Exception as e:
    logger.warning("Could not load contacts.json: %s", e)


def _extract_recipient_and_message(text: str) -> tuple[str, str]:
    """
    Parse intent text like:
    "Send a WhatsApp message to Raj saying I'll be late"
    "Message mom that I'm home"
    Returns (recipient_name, message_body)
    """
    # Pattern: "to <name> (saying|that|:) <message>"
    match = re.search(
        r"(?:whatsapp\s+message\s+to|message\s+to|send\s+to|to)\s+(\w+)\s+(?:saying|that|:)?\s*(.+)",
        text, re.IGNORECASE
    )
    if match:
        return match.group(1).strip().lower(), match.group(2).strip()
    return "", text


class WhatsAppAction:
    async def handle(self, text: str) -> str:
        name, message = _extract_recipient_and_message(text)

        if not name:
            return "Who do you want to message? Say something like: message Raj saying I'll be late."

        phone = CONTACTS.get(name)
        if not phone:
            return (
                f"I don't have {name}'s number saved. "
                f"Add them to data/contacts.json with their full number including country code."
            )

        if not message:
            return "What should I say in the message?"

        logger.info("Sending WhatsApp to %s (%s): %s", name, phone, message)

        try:
            script_path = os.path.join(os.path.dirname(__file__), "whatsapp_service", "send.js")
            process = await asyncio.create_subprocess_exec(
                "node", script_path, phone, message,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return f"Message sent to {name}."
            else:
                logger.error("WhatsApp Baileys send failed: %s", stderr.decode())
                return f"Failed to send message to {name}."
        except Exception as e:
            logger.error("WhatsApp send failed: %s", e)
            return f"Failed to send message to {name}. Make sure Node.js is installed."
