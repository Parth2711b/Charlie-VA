import pytest
import asyncio
import sys
from unittest.mock import AsyncMock, patch, MagicMock

# Mock chromadb since it might not be installed in the test env
sys.modules['chromadb'] = MagicMock()

# Import the modules we want to test
from core.intent_router import IntentRouter
from core.assistant import Assistant


@pytest.mark.asyncio
async def test_theme_switching_intent():
    """
    Test that saying a theme command correctly triggers the websocket broadcast
    and returns the right confirmation message.
    """
    with patch('core.websocket_bridge.emit', new_callable=AsyncMock) as mock_emit:
        
        router = IntentRouter()
        context = []
        response = await router.route("switch to hacker theme", context)
        
        assert "Switching to Hacker theme" in response
        mock_emit.assert_called_with({"type": "set_theme", "theme": "hacker"})


@pytest.mark.asyncio
async def test_barge_in_flag_logic():
    """
    Test that the assistant correctly sets the barge_in_triggered flag
    if the wake word is detected while speaking.
    """
    with patch('speech.wake_word.WakeWordDetector') as mock_wake, \
         patch('speech.stt.STT') as mock_stt, \
         patch('speech.tts.TTS') as mock_tts, \
         patch('core.memory.Memory') as mock_mem:
        
        assistant = Assistant()
        
        async def mock_speak(text):
            return 2.0
        assistant._speak = mock_speak
        
        assistant.wake_word.wait_for_wake_word = MagicMock()
        
        # FIX: Make ws an AsyncMock so we can await its methods
        assistant.ws = AsyncMock()
        assistant.tts = MagicMock()
        
        assistant.router.route = AsyncMock(return_value="Dummy response")
        
        # Actually trigger the barge-in handler!
        assistant._handle_barge_in()
        
        assert assistant.barge_in_triggered == True
        assistant.ws.emit.assert_called_with({"type": "barge_in_ack"})
