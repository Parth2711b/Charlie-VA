"""
actions/system.py — System-level actions: volume, apps, clipboard.
Windows-focused (since Parth's machine is Windows 11).
"""

import logging
import subprocess
import platform
import asyncio

logger = logging.getLogger("Charlie.system")

SYSTEM = platform.system()


class SystemAction:
    async def handle(self, text: str) -> str:
        text_lower = text.lower()

        if "volume up" in text_lower or "increase volume" in text_lower:
            return self._volume("up")

        elif "volume down" in text_lower or "decrease volume" in text_lower:
            return self._volume("down")

        elif "mute" in text_lower:
            return self._mute()

        elif "open" in text_lower:
            app = self._extract_app(text_lower)
            return self._open_app(app) if app else "Which app should I open?"

        elif "clipboard" in text_lower or "copy" in text_lower:
            return self._read_clipboard()

        else:
            return "I'm not sure what system action you want. Try: volume up, open notepad, read clipboard."

    def _volume(self, direction: str) -> str:
        if SYSTEM != "Windows":
            return "Volume control only supported on Windows right now."
        try:
            # Uses nircmd (optional) or PowerShell
            step = 2000 if direction == "up" else -2000
            subprocess.run(
                ["powershell", "-c",
                 f"(New-Object -ComObject WScript.Shell).SendKeys([char]{'175' if direction == 'up' else '174'})"],
                check=True, capture_output=True
            )
            return f"Volume {'increased' if direction == 'up' else 'decreased'}."
        except Exception as e:
            logger.error("Volume error: %s", e)
            return "Couldn't change volume."

    def _mute(self) -> str:
        try:
            subprocess.run(
                ["powershell", "-c",
                 "(New-Object -ComObject WScript.Shell).SendKeys([char]173)"],
                check=True, capture_output=True
            )
            return "Muted."
        except Exception as e:
            logger.error("Mute error: %s", e)
            return "Couldn't mute."

    def _open_app(self, app: str) -> str:
        COMMON_APPS = {
            "notepad":      "notepad.exe",
            "calculator":   "calc.exe",
            "paint":        "mspaint.exe",
            "task manager": "taskmgr.exe",
            "file explorer":"explorer.exe",
            "vs code":      "code",
            "chrome":       "chrome",
            "terminal":     "wt",
        }
        exe = COMMON_APPS.get(app, app)
        try:
            subprocess.Popen([exe])
            return f"Opening {app}."
        except FileNotFoundError:
            return f"Couldn't find {app}. Make sure it's installed and in PATH."

    def _extract_app(self, text: str) -> str:
        import re
        match = re.search(r"open\s+(.+)", text)
        return match.group(1).strip() if match else ""

    def _read_clipboard(self) -> str:
        try:
            import pyperclip
            content = pyperclip.paste()
            if content:
                return f"Clipboard contains: {content[:200]}"
            return "Clipboard is empty."
        except Exception as e:
            logger.error("Clipboard error: %s", e)
            return "Couldn't read clipboard."
