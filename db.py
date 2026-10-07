"""SQLite database for storing pronunciation practice history."""
import json
import os
import sqlite3
from datetime import datetime

DB_PATH = os.getenv("PRACTICE_DB_PATH") or os.path.join(os.path.dirname(__file__), "practice_history.db")
SAMPLE_HISTORY_PATH = os.path.join(os.path.dirname(__file__), "sample_history.json")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT NOT NULL,
            target_phonemes TEXT NOT NULL,
            predicted_phonemes TEXT NOT NULL,
            score REAL NOT NULL,
            wer REAL NOT NULL,
            errors TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'real'
        )
    """)

    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(attempts)").fetchall()
    }
    if "source" not in columns:
        conn.execute("ALTER TABLE attempts ADD COLUMN source TEXT NOT NULL DEFAULT 'real'")

    conn.commit()
    conn.close()


def seed_history_if_empty():
    conn = get_connection()
    row = conn.execute("SELECT COUNT(*) AS count FROM attempts").fetchone()
    attempts_count = row["count"]

    if attempts_count > 0:
        conn.close()
        return

    if not os.path.exists(SAMPLE_HISTORY_PATH):
        conn.close()
        return

    with open(SAMPLE_HISTORY_PATH) as f:
        sample_attempts = json.load(f)

    for attempt in sample_attempts:
        conn.execute(
            """INSERT INTO attempts (word, target_phonemes, predicted_phonemes, score, wer, errors, timestamp, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                attempt["word"],
                json.dumps(attempt["target_phonemes"]),
                json.dumps(attempt["predicted_phonemes"]),
                attempt["score"],
                attempt["wer"],
                json.dumps(attempt["errors"]),
                attempt["timestamp"],
                "sample",
            ),
        )

    conn.commit()
    conn.close()


def save_attempt(word, target_phonemes, predicted_phonemes, score, wer, errors, source="real"):
    conn = get_connection()
    conn.execute(
        """INSERT INTO attempts (word, target_phonemes, predicted_phonemes, score, wer, errors, timestamp, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            word,
            json.dumps(target_phonemes),
            json.dumps(predicted_phonemes),
            score,
            wer,
            json.dumps(errors),
            datetime.now().isoformat(),
            source,
        ),
    )
    conn.commit()
    conn.close()


def get_history(limit=50):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM attempts ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    results = []
    for row in rows:
        results.append({
            "id": row["id"],
            "word": row["word"],
            "target_phonemes": json.loads(row["target_phonemes"]),
            "predicted_phonemes": json.loads(row["predicted_phonemes"]),
            "score": row["score"],
            "wer": row["wer"],
            "errors": json.loads(row["errors"]),
            "timestamp": row["timestamp"],
            "source": row["source"],
        })
    return results


def get_today_count():
    today = datetime.now().date().isoformat()
    conn = get_connection()
    row = conn.execute(
        "SELECT COUNT(*) AS count FROM attempts WHERE source='real' AND timestamp LIKE ?",
        (f"{today}%",),
    ).fetchone()
    conn.close()
    return row["count"]


def clear_history(source: str | None = None):
    conn = get_connection()
    if source:
        conn.execute("DELETE FROM attempts WHERE source = ?", (source,))
    else:
        conn.execute("DELETE FROM attempts")
    conn.commit()
    conn.close()


# Initialize on import
init_db()
seed_history_if_empty()
