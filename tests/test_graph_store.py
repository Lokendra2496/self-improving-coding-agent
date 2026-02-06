from memory.graph_store import LocalGraphStore


def test_local_graph_store_appends() -> None:
    store = LocalGraphStore()
    store.upsert_episode({"run_id": "r1", "iteration": 0, "steps": []})
    store.upsert_episode({"run_id": "r1", "iteration": 1, "steps": []})
    assert len(store.episodes) == 2
