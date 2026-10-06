import psycopg
from psycopg.types.json import Jsonb


def create(conn: psycopg.Connection, title: str) -> int:
    return conn.execute("INSERT INTO conversations (title) VALUES (%s) RETURNING id", (title,)).fetchone()[0]


def add(conn: psycopg.Connection, cid: int, role: str, content: str, payload: dict) -> None:
    conn.execute("INSERT INTO messages (conversation_id, role, content, payload) VALUES (%s, %s, %s, %s)",
                 (cid, role, content, Jsonb(payload)))
    conn.execute("UPDATE conversations SET updated_at = now() WHERE id = %s", (cid,))


def messages(conn: psycopg.Connection, cid: int) -> list[dict] | None:
    """A conversation's messages in order, or None if it doesn't exist."""
    if not conn.execute("SELECT 1 FROM conversations WHERE id = %s", (cid,)).fetchone():
        return None
    rows = conn.execute("SELECT role, content, payload FROM messages WHERE conversation_id = %s ORDER BY id",
                        (cid,)).fetchall()
    return [{"role": r, "content": c, "payload": p} for r, c, p in rows]


def conversations(conn: psycopg.Connection) -> list[dict]:
    rows = conn.execute("SELECT id, title, updated_at FROM conversations ORDER BY updated_at DESC").fetchall()
    return [{"id": i, "title": t, "updated_at": u.isoformat()} for i, t, u in rows]


def title(conn: psycopg.Connection, cid: int) -> str | None:
    row = conn.execute("SELECT title FROM conversations WHERE id = %s", (cid,)).fetchone()
    return row[0] if row else None
