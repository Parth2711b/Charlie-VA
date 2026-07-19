import pytest
from core.intent_router import _is_shutdown, _needs_live_data, IntentRouter

def test_is_shutdown():
    assert _is_shutdown("shut down charlie") is True
    assert _is_shutdown("goodbye charlie") is True
    assert _is_shutdown("hello charlie") is False

def test_semantic_match():
    router = IntentRouter()
    
    # Exact keyword matches should still map perfectly (distance 0)
    assert router._semantic_match("open calculator") == "calculator"
    
    # Semantic variations that ARE NOT in the exact keywords list should also map!
    # "how hot is it" -> weather
    assert router._semantic_match("is it going to rain") == "weather"
    assert router._semantic_match("put on some jazz") == "music"

def test_needs_live_data_heuristics():
    # Should trigger web search
    assert _needs_live_data("what is the latest news") is True
    assert _needs_live_data("who won the match today") is True
    assert _needs_live_data("what is the current stock price of Apple") is True
    
    # Should NOT trigger web search (pure offline LLM)
    assert _needs_live_data("write a poem about a cat") is False
    assert _needs_live_data("what is the definition of artificial intelligence") is False
    assert _needs_live_data("who are you") is False

@pytest.mark.asyncio
async def test_dashboard_command_map_fallback():
    """Test that dashboard map routing falls back properly if context is missing."""
    router = IntentRouter(memory=None)
    
    # We shouldn't crash if the command is for the dashboard
    class MockWS:
        def __init__(self):
            self.emitted = []
        async def emit(self, message: dict, target_user_id: int | None = None):
            self.emitted.append(message)
            
    # Swap out the real ws with a mock
    from core import websocket_bridge
    original_emit = websocket_bridge.emit
    
    mock_ws = MockWS()
    websocket_bridge.emit = mock_ws.emit
    
    try:
        res = await router._handle_dashboard_command("show me the map")
        assert "map" in res.lower() or "focus" in res.lower()
        
        # Verify an event was emitted to focus the map
        assert any(e.get("type") == "focus_panel" and e.get("panel") == "map" for e in mock_ws.emitted)
    finally:
        websocket_bridge.emit = original_emit
