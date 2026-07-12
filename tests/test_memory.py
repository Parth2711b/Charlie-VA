import pytest
import os
from core.memory import Memory

@pytest.fixture
def memory_db(tmp_path):
    # Use a temporary database for testing
    import config
    original_path = config.MEMORY_DB_PATH
    temp_db_path = tmp_path / "test_memory.db"
    
    # We patch the instance directly in the test to use the temp path
    mem = Memory()
    mem.db_path = str(temp_db_path)
    mem._init_db() # re-init on the new path
    
    yield mem
    
    # Cleanup (fixture teardown)
    pass

def test_memory_add_turn(memory_db):
    memory_db.add_turn("hello", "hi there")
    ctx = memory_db.get_context()
    assert len(ctx) == 2
    assert ctx[0]["role"] == "user"
    assert ctx[0]["content"] == "hello"
    assert ctx[1]["role"] == "assistant"
    assert ctx[1]["content"] == "hi there"

def test_memory_context_limit(memory_db):
    import core.memory
    original_limit = core.memory.MAX_CONTEXT_TURNS
    core.memory.MAX_CONTEXT_TURNS = 2
    
    memory_db.add_turn("1", "1")
    memory_db.add_turn("2", "2")
    memory_db.add_turn("3", "3")
    
    ctx = memory_db.get_context()
    # 2 turns * 2 (user+assistant) = 4 items
    assert len(ctx) == 4

    
    # Should only contain turn 2 and 3
    assert ctx[0]["content"] == "2"
    assert ctx[1]["content"] == "2"
    assert ctx[2]["content"] == "3"
    assert ctx[-1]["content"] == "3"
    
    core.memory.MAX_CONTEXT_TURNS = original_limit

def test_memory_save_and_get_fact(memory_db):
    memory_db.save_fact("name", "John")
    assert memory_db.get_fact("name") == "John"
    
    # Test update
    memory_db.save_fact("name", "Jane")
    assert memory_db.get_fact("name") == "Jane"
    
    assert memory_db.get_fact("nonexistent") is None

def test_memory_clear_context(memory_db):
    memory_db.add_turn("hello", "hi")
    memory_db.save_fact("color", "blue")
    
    memory_db.clear_context()
    
    # Context should be empty
    assert len(memory_db.get_context()) == 0
    
    # Facts should still exist
    assert memory_db.get_fact("color") == "blue"
