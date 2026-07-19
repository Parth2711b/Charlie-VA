import pytest
import contextvars
from core import websocket_bridge

@pytest.fixture(autouse=True)
def reset_state():
    """Reset the clients and context var before each test"""
    websocket_bridge._clients.clear()
    websocket_bridge.current_user_id.set(None)
    yield
    websocket_bridge._clients.clear()
    websocket_bridge.current_user_id.set(None)


def test_has_clients_no_context_fallback():
    """Test that when current_user_id is not set, it defaults to user 1 for local microphone commands."""
    # Simulate a connected dashboard for user 1
    websocket_bridge._clients[1] = {object()}
    
    # Context is None
    assert websocket_bridge.current_user_id.get() is None
    
    # Should fallback to user 1 and find the client
    assert websocket_bridge.has_clients() is True


def test_has_clients_with_context():
    """Test that when current_user_id IS set, it correctly checks that specific user's clients."""
    # Simulate dashboard for user 2
    websocket_bridge._clients[2] = {object()}
    
    websocket_bridge.current_user_id.set(2)
    assert websocket_bridge.has_clients() is True
    
    websocket_bridge.current_user_id.set(3)
    assert websocket_bridge.has_clients() is False


@pytest.mark.asyncio
async def test_emit_fallback():
    """Test that emit falls back to user 1 if no context is set."""
    class MockWebsocket:
        def __init__(self):
            self.sent = []
        async def send(self, data):
            self.sent.append(data)
            
    ws_client = MockWebsocket()
    websocket_bridge._clients[1] = {ws_client}
    
    # Context is None. Should emit to User 1
    assert websocket_bridge.current_user_id.get() is None
    await websocket_bridge.emit({"test": "data"})
    
    assert len(ws_client.sent) == 1
    assert "test" in ws_client.sent[0]
