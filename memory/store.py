"""SQLite 长期记忆：用户画像事实 + 对话存档。

每次操作使用独立短连接，避免多线程/多进程（Web 服务）下的连接共享问题。
"""
import sqlite3
import time

import config


def _conn() -> sqlite3.Connection:
    config.ensure_dirs()
    conn = sqlite3.connect(str(config.MEMORY_DB))
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS profile_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact TEXT NOT NULL,
            source TEXT,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT, answer TEXT, image TEXT,
            violations TEXT, created_at TEXT
        );
        """
    )
    return conn


def add_fact(fact: str, source: str = "user_statement") -> None:
    fact = (fact or "").strip()
    if not fact:
        return
    with _conn() as conn:
        conn.execute(
            "INSERT INTO profile_facts (fact, source, created_at) VALUES (?, ?, ?)",
            (fact, source, time.strftime("%Y-%m-%d %H:%M:%S")),
        )


def get_facts(limit: int = 50) -> list:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT fact, source, created_at FROM profile_facts ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def log_turn(question: str, answer: str, image: str, violations: str) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO conversations (question, answer, image, violations, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (question, answer, image or "", violations or "[]",
             time.strftime("%Y-%m-%d %H:%M:%S")),
        )


def recent_turns(limit: int = 20) -> list:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT question, answer, image, violations, created_at FROM conversations "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
