from agent.state import RunConfig
from memory.store import MemoryStore


def test_mem0_memory_store_round_trip() -> None:
    config = RunConfig(goal="memory test")
    store = MemoryStore.from_config(config)
    store.store_episode({"summary": "first"})
    results = store.get_similar("query", top_k=1)
    assert results == []
