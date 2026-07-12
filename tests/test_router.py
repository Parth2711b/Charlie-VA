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
