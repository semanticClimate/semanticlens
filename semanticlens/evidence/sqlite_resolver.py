"""
Persistent source evidence storage for SemanticLens

SQLite stores EvidenceUnit metadata, not vectors or page image bytes

The resolver follows Core's EvidenceResolver contract and preserves the
retriever's result order, scores and labels
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType

from semanticlens.core.exceptions import ConfigurationError, EvidenceResolutionError
from semanticlens.core.schemas import EvidenceUnit, RetrievalCandidate, SearchHit


_SCHEMA_VERSION = 1
_LOOKUP_BATCH_SIZE = 500 


class SQLiteEvidenceResolver:
    """
    Persist and resolve evidence metadata within a shared workspace.

    Data lives at ``workspace/evidence/evidence.sqlite3``. Use the same
    workspace for the document processor, embedder, and evidence resolver.

    Each instance owns one SQLite connection; use it sequentially from the
    thread that created it. Separate instances may reopen the same database.
    """

    def __init__(self, *, workspace: str | Path) -> None:
        if not isinstance(workspace, (str, Path)) or (
            isinstance(workspace, str) and not workspace.strip()
        ):
            raise ConfigurationError("workspace must be a non-empty path")

        try:
            self.workspace = Path(workspace).expanduser().resolve()
            directory = self.workspace / "evidence"
            directory.mkdir(parents=True, exist_ok=True)
        except (OSError, RuntimeError) as exc:
            raise ConfigurationError("Cannot prepare evidence workspace") from exc

        self.database_path = directory / "evidence.sqlite3"
        self._connection: sqlite3.Connection | None = None
        try:
            self._connection = sqlite3.connect(self.database_path, timeout=10.0)
            self._initialize_schema()
        except Exception as exc:
            if self._connection is not None:
                try:
                    self._connection.close()
                except sqlite3.Error:
                    pass
                self._connection = None
            if isinstance(exc, sqlite3.Error):
                raise EvidenceResolutionError("Failed to open evidence database") from exc
            raise

    def add(self, units: Sequence[EvidenceUnit]) -> None:
        """Atomically insert or replace evidence units by their unit IDs"""
        connection = self._require_open()
        if not isinstance(units, Sequence) or isinstance(units, (str, bytes)):
            raise TypeError("units must be a sequence of EvidenceUnit objects")

        rows: list[tuple[str, str]] = []
        seen_ids: set[str] = set()
        for unit in units:
            if not isinstance(unit, EvidenceUnit):
                raise TypeError("units must contain EvidenceUnit objects")
            if unit.unit_id in seen_ids:
                raise ValueError(f"Duplicate evidence unit ID in batch: {unit.unit_id!r}")
            seen_ids.add(unit.unit_id)
            rows.append((unit.unit_id, unit.model_dump_json()))

        if not rows:
            return

        try:
            with connection:
                connection.executemany(
                    """
                    INSERT INTO evidence_units (unit_id, evidence_json)
                    VALUES (?, ?)
                    ON CONFLICT(unit_id) DO UPDATE SET
                        evidence_json = excluded.evidence_json
                    """,
                    rows,
                )
        except sqlite3.Error as exc:
            raise EvidenceResolutionError("Failed to store evidence metadata") from exc

    def resolve(self, candidates: Sequence[RetrievalCandidate]) -> list[SearchHit]:
        """Resolve every candidate without changing its order, score or label"""
        connection = self._require_open()
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            raise TypeError("candidates must be a sequence of RetrievalCandidate objects")

        ids: list[str] = []
        seen_ids: set[str] = set()
        for candidate in candidates:
            if not isinstance(candidate, RetrievalCandidate):
                raise TypeError("candidates must contain RetrievalCandidate objects")
            if candidate.unit_id in seen_ids:
                raise ValueError(f"Duplicate retrieval candidate ID: {candidate.unit_id!r}")
            seen_ids.add(candidate.unit_id)
            ids.append(candidate.unit_id)

        if not ids:
            return []

        records: dict[str, str] = {}
        try:
            for start in range(0, len(ids), _LOOKUP_BATCH_SIZE):
                batch = ids[start : start + _LOOKUP_BATCH_SIZE]
                placeholders = ", ".join("?" for _ in batch)
                rows = connection.execute(
                    "SELECT unit_id, evidence_json FROM evidence_units "
                    f"WHERE unit_id IN ({placeholders})",
                    batch,
                ).fetchall()
                records.update(rows)
        except sqlite3.Error as exc:
            raise EvidenceResolutionError("Failed to load evidence metadata") from exc

        hits: list[SearchHit] = []
        for candidate in candidates:
            saved_json = records.get(candidate.unit_id)
            if saved_json is None:
                raise EvidenceResolutionError(
                    f"Evidence not found for unit ID {candidate.unit_id!r}. "
                    "Check the workspace or re-index the source document."
                )
            try:
                evidence = EvidenceUnit.model_validate_json(saved_json)
            except (ValueError, TypeError) as exc:
                raise EvidenceResolutionError(
                    f"Stored evidence is invalid for unit ID {candidate.unit_id!r}"
                ) from exc
            if evidence.unit_id != candidate.unit_id:
                raise EvidenceResolutionError(
                    f"Stored evidence ID does not match its key: {candidate.unit_id!r}"
                )
            hits.append(
                SearchHit(
                    evidence=evidence,
                    score=candidate.score,
                    retriever=candidate.retriever,
                )
            )
        return hits

    def close(self) -> None:
        """Close the SQLite connection; safe to call more than once"""
        connection, self._connection = self._connection, None
        if connection is not None:
            try:
                connection.close()
            except sqlite3.Error as exc:
                raise EvidenceResolutionError("Failed to close evidence database") from exc

    def __enter__(self) -> SQLiteEvidenceResolver:
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _require_open(self) -> sqlite3.Connection:
        if self._connection is None:
            raise EvidenceResolutionError("This evidence resolver is closed")
        return self._connection

    def _initialize_schema(self) -> None:
        """Create only an empty database; never silently migrate unknown data"""
        connection = self._require_open()
        try:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version == 0:
                existing = connection.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE substr(name, 1, 7) != 'sqlite_' "
                    "AND type IN ('table', 'view', 'index', 'trigger')"
                ).fetchone()
                if existing is not None:
                    raise EvidenceResolutionError(
                        "Existing evidence database has no recognized schema version"
                    )
                with connection:
                    connection.execute(
                        """
                        CREATE TABLE evidence_units (
                            unit_id TEXT PRIMARY KEY NOT NULL,
                            evidence_json TEXT NOT NULL
                        )
                        """
                    )
                    connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            elif version != _SCHEMA_VERSION:
                raise EvidenceResolutionError(
                    f"Unsupported evidence database schema version: {version}"
                )

            columns = connection.execute("PRAGMA table_info(evidence_units)").fetchall()
            expected = [
                ("unit_id", "TEXT", 1, 1),
                ("evidence_json", "TEXT", 1, 0),
            ]
            actual = [(col[1], col[2].upper(), col[3], col[5]) for col in columns]
            if actual != expected:
                raise EvidenceResolutionError("Evidence database has an incompatible schema")
        except sqlite3.Error as exc:
            raise EvidenceResolutionError("Failed to initialize evidence database") from exc
