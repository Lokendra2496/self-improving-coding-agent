from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase

from memory.graph_config import GraphConfig, resolve_graph_config


class GraphStore:
    def upsert_episode(self, episode: Dict[str, Any]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None

    @classmethod
    def from_config(cls, config: GraphConfig) -> "GraphStore":
        if config.backend != "neo4j":
            raise ValueError(f"Unsupported graph backend: {config.backend}")
        return Neo4jGraphStore.from_config(config)


class LocalGraphStore(GraphStore):
    def __init__(self) -> None:
        self.episodes: List[Dict[str, Any]] = []

    def upsert_episode(self, episode: Dict[str, Any]) -> None:
        self.episodes.append(episode)


@dataclass
class Neo4jGraphStore(GraphStore):
    driver: Any
    database: Optional[str]

    @classmethod
    def from_config(cls, config: GraphConfig) -> "Neo4jGraphStore":
        if not config.uri or not config.user or not config.password:
            raise ValueError("NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD are required.")
        driver = GraphDatabase.driver(config.uri, auth=(config.user, config.password))
        return cls(driver=driver, database=config.database)

    def upsert_episode(self, episode: Dict[str, Any]) -> None:
        with self.driver.session(database=self.database) as session:
            session.execute_write(self._upsert_tx, episode)

    def close(self) -> None:
        self.driver.close()

    @staticmethod
    def _upsert_tx(tx, episode: Dict[str, Any]) -> None:
        run_id = episode.get("run_id")
        goal = episode.get("goal")
        status = episode.get("status")
        iteration = episode.get("iteration")
        steps = episode.get("steps", [])

        tx.run(
            """
            MERGE (r:Run {run_id: $run_id})
            ON CREATE SET r.goal = $goal
            SET r.status = $status
            """,
            run_id=run_id,
            goal=goal,
            status=status,
        )

        tx.run(
            """
            MERGE (i:Iteration {run_id: $run_id, index: $index})
            WITH i
            MATCH (r:Run {run_id: $run_id})
            MERGE (r)-[:HAS_ITERATION]->(i)
            """,
            run_id=run_id,
            index=iteration,
        )

        for step in steps:
            details = step.get("details", {})
            details_json = json.dumps(details, ensure_ascii=True, default=str)
            tx.run(
                """
                MERGE (s:Step {run_id: $run_id, iteration: $iteration, name: $name, idx: $idx})
                SET s.status = $status,
                    s.details = $details,
                    s.started_at = $started_at,
                    s.ended_at = $ended_at
                WITH s
                MATCH (i:Iteration {run_id: $run_id, index: $iteration})
                MERGE (i)-[:HAS_STEP]->(s)
                """,
                run_id=run_id,
                iteration=iteration,
                name=step.get("name"),
                idx=step.get("idx"),
                status=step.get("status"),
                details=details_json,
                started_at=step.get("started_at"),
                ended_at=step.get("ended_at"),
            )

        for path in episode.get("retrieved_files", []):
            tx.run(
                """
                MERGE (f:File {path: $path})
                WITH f
                MATCH (i:Iteration {run_id: $run_id, index: $iteration})
                MERGE (i)-[:RETRIEVED_FILE]->(f)
                """,
                path=path,
                run_id=run_id,
                iteration=iteration,
            )

        if episode.get("failure_signature"):
            tx.run(
                """
                MERGE (fs:FailureSignature {id: $id})
                SET fs.text = $text
                WITH fs
                MATCH (i:Iteration {run_id: $run_id, index: $iteration})
                MERGE (i)-[:HAS_FAILURE]->(fs)
                """,
                id=episode["failure_signature"].get("id"),
                text=episode["failure_signature"].get("text"),
                run_id=run_id,
                iteration=iteration,
            )

        if episode.get("bug_type"):
            tx.run(
                """
                MERGE (b:BugType {name: $name})
                WITH b
                MATCH (i:Iteration {run_id: $run_id, index: $iteration})
                MERGE (i)-[:CLASSIFIED_AS]->(b)
                """,
                name=episode["bug_type"],
                run_id=run_id,
                iteration=iteration,
            )

        fix_card = episode.get("fix_card")
        if isinstance(fix_card, dict):
            fix_id = fix_card.get("id") or f"{run_id}:{iteration}"
            files_changed = fix_card.get("files_changed") or []
            if not isinstance(files_changed, list):
                files_changed = []
            tx.run(
                """
                MERGE (fc:FixCard {id: $id})
                SET fc.summary = $summary,
                    fc.root_cause = $root_cause,
                    fc.fix = $fix,
                    fc.verification = $verification,
                    fc.error_signature = $error_signature,
                    fc.status = $status,
                    fc.repo_path = $repo_path,
                    fc.goal = $goal,
                    fc.run_id = $run_id,
                    fc.iteration = $iteration
                WITH fc
                MATCH (r:Run {run_id: $run_id})
                MERGE (r)-[:HAS_FIX_CARD]->(fc)
                """,
                id=fix_id,
                summary=fix_card.get("summary"),
                root_cause=fix_card.get("root_cause"),
                fix=fix_card.get("fix"),
                verification=fix_card.get("verification"),
                error_signature=fix_card.get("error_signature"),
                status=fix_card.get("status"),
                repo_path=fix_card.get("repo_path"),
                goal=fix_card.get("goal"),
                run_id=run_id,
                iteration=iteration,
            )

            for path in files_changed:
                if not isinstance(path, str):
                    continue
                tx.run(
                    """
                    MERGE (f:File {path: $path})
                    WITH f
                    MATCH (fc:FixCard {id: $id})
                    MERGE (fc)-[:CHANGED_FILE]->(f)
                    """,
                    path=path,
                    id=fix_id,
                )

            if fix_card.get("error_signature"):
                tx.run(
                    """
                    MERGE (fs:FailureSignature {id: $id})
                    SET fs.text = $text
                    WITH fs
                    MATCH (fc:FixCard {id: $fix_id})
                    MERGE (fc)-[:HAS_FAILURE]->(fs)
                    """,
                    id=fix_card.get("error_signature"),
                    text=fix_card.get("error_signature"),
                    fix_id=fix_id,
                )
