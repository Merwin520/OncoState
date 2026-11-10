"""Append-only correction-event log.

correction_events.jsonl is the source of truth: one sealed event per line, each line carrying the SHA-256 of the
previous event (hash chain), so any edit or deletion is detectable with verify(). An SQLite mirror and a CSV export
are derived views and can always be rebuilt from the JSONL.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl  # Linux / macOS
except ImportError:  # Windows
    fcntl = None
    import msvcrt

import pandas as pd

from .correction_event import seal
from state.provenance import sha256_json


class TamperedLogError(RuntimeError):
    pass


@contextmanager
def _exclusive(f, lock_path: Path):
    """Exclusive lock so two reviewers writing at once cannot interleave lines (flock on POSIX, msvcrt on Windows)."""
    if fcntl is not None:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
        return
    with open(lock_path, "a+b") as lock:
        lock.seek(0)
        while True:
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                time.sleep(0.05)
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


class EventStore:
    def __init__(self, root: str | Path, sqlite_mirror: bool = True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "correction_events.jsonl"
        self.path.touch(exist_ok=True)
        self.db = self.root / "correction_events.sqlite" if sqlite_mirror else None

    def all(self) -> list[dict]:
        return [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def for_patient(self, pid: str) -> list[dict]:
        return [e for e in self.all() if e["patient_id"] == pid]

    def append(self, event: dict) -> dict:
        with open(self.path, "a+", encoding="utf-8", newline="") as f, _exclusive(f, self.root / ".events.lock"):
            f.seek(0)
            lines = [l for l in f.read().splitlines() if l.strip()]
            prev = json.loads(lines[-1])["event_sha256"] if lines else None
            sealed = seal(event, prev)
            f.write(json.dumps(sealed, sort_keys=True) + "\n")
            f.flush()
        if self.db:
            self._mirror(sealed)
        return sealed

    def verify(self) -> int:
        prev = None
        events = self.all()
        for i, e in enumerate(events):
            if e.get("prev_event_sha256") != prev:
                raise TamperedLogError(f"hash chain broken at line {i + 1}")
            if sha256_json({k: v for k, v in e.items() if k != "event_sha256"}) != e["event_sha256"]:
                raise TamperedLogError(f"event modified at line {i + 1}")
            prev = e["event_sha256"]
        return len(events)

    def sha256(self) -> str:
        return sha256_json(self.all())

    def _mirror(self, e: dict) -> None:
        # Optional derived view, rebuildable from the JSONL. A broken sqlite build (e.g. a libstdc++ ABI mismatch on
        # import) must never interrupt a clinician decision that has already been appended to the source of truth.
        try:
            import sqlite3
            self._mirror_sqlite(sqlite3, e)
        except Exception:  # noqa: BLE001
            self.db = None

    def _mirror_sqlite(self, sqlite3, e: dict) -> None:
        with sqlite3.connect(self.db) as con:
            con.execute("CREATE TABLE IF NOT EXISTS events (event_id TEXT PRIMARY KEY, patient_id TEXT, batch_id TEXT, "
                        "correction_source TEXT, event_type TEXT, action TEXT, timestamp TEXT, json TEXT)")
            con.execute("INSERT OR IGNORE INTO events VALUES (?,?,?,?,?,?,?,?)",
                        (e["event_id"], e["patient_id"], e["batch_id"], e["correction_source"], e["event_type"], e["action"],
                         e["timestamp"], json.dumps(e, sort_keys=True)))

    def export_csv(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else self.root / "correction_events.csv"
        rows = self.all()
        df = pd.DataFrame(rows)
        for c in ("original_prediction", "corrected_value", "registration_quality"):
            if c in df:
                df[c] = df[c].map(lambda v: json.dumps(v) if isinstance(v, (dict, list)) else v)
        df.to_csv(path, index=False)
        return path
