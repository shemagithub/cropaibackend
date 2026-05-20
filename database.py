"""MySQL async database layer (replaces MongoDB/Motor)."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import aiomysql

logger = logging.getLogger("cropdoctor.db")
_pool: Optional[aiomysql.Pool] = None


def mysql_settings() -> dict:
    return {
        "host": os.environ.get("MYSQL_HOST", "localhost"),
        "port": int(os.environ.get("MYSQL_PORT", "3306")),
        "user": os.environ["MYSQL_USER"],
        "password": os.environ.get("MYSQL_PASSWORD", ""),
        "db": os.environ["MYSQL_DATABASE"],
    }


def _utc(dt: Any) -> Any:
    if isinstance(dt, datetime) and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _json_dumps(value: Any) -> str:
    return json.dumps(value, default=str)


def _json_loads(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    return json.loads(value)


async def ensure_database() -> None:
    cfg = mysql_settings()
    db_name = cfg.pop("db")
    conn = await aiomysql.connect(**cfg)
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
    finally:
        conn.close()


async def init_pool() -> None:
    global _pool
    for key in ("MYSQL_USER", "MYSQL_DATABASE"):
        if not os.environ.get(key):
            raise RuntimeError(f"Missing required env var: {key}")
    await ensure_database()
    cfg = mysql_settings()
    _pool = await aiomysql.create_pool(
        host=cfg["host"],
        port=cfg["port"],
        user=cfg["user"],
        password=cfg["password"],
        db=cfg["db"],
        autocommit=True,
        minsize=1,
        maxsize=10,
        charset="utf8mb4",
    )
    await init_schema()
    logger.info("MySQL connected: %s@%s:%s/%s", cfg["user"], cfg["host"], cfg["port"], cfg["db"])


async def close_pool() -> None:
    global _pool
    if _pool:
        _pool.close()
        await _pool.wait_closed()
        _pool = None


async def init_schema() -> None:
    async with _pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id VARCHAR(64) PRIMARY KEY,
                    email VARCHAR(255) NOT NULL UNIQUE,
                    name VARCHAR(255) NOT NULL,
                    phone VARCHAR(64) NULL,
                    role VARCHAR(32) NOT NULL DEFAULT 'farmer',
                    language VARCHAR(16) NOT NULL DEFAULT 'en',
                    picture TEXT NULL,
                    location VARCHAR(255) NULL,
                    password VARCHAR(255) NULL,
                    created_at DATETIME(6) NOT NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            await cur.execute(
                """
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_token VARCHAR(512) PRIMARY KEY,
                    user_id VARCHAR(64) NOT NULL,
                    expires_at DATETIME(6) NOT NULL,
                    created_at DATETIME(6) NOT NULL,
                    INDEX idx_user_sessions_user (user_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            await cur.execute(
                """
                CREATE TABLE IF NOT EXISTS scans (
                    scan_id VARCHAR(64) PRIMARY KEY,
                    user_id VARCHAR(64) NOT NULL,
                    crop_name VARCHAR(255) NULL,
                    image_base64 LONGTEXT NOT NULL,
                    disease_name VARCHAR(255) NOT NULL,
                    is_healthy TINYINT(1) NOT NULL,
                    severity VARCHAR(32) NOT NULL,
                    confidence DOUBLE NOT NULL,
                    symptoms TEXT NOT NULL,
                    causes TEXT NOT NULL,
                    affected_part VARCHAR(255) NOT NULL,
                    spread_risk VARCHAR(32) NOT NULL,
                    next_actions JSON NOT NULL,
                    treatments JSON NOT NULL,
                    preventive_measures JSON NOT NULL,
                    notes TEXT NULL,
                    created_at DATETIME(6) NOT NULL,
                    INDEX idx_scans_user (user_id),
                    INDEX idx_scans_created (created_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            await cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chatbot_msgs (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(128) NOT NULL,
                    user_id VARCHAR(64) NOT NULL,
                    role VARCHAR(32) NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME(6) NOT NULL,
                    INDEX idx_chat_session (session_id),
                    INDEX idx_chat_user (user_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            await cur.execute(
                """
                CREATE TABLE IF NOT EXISTS expert_messages (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    thread_id VARCHAR(128) NOT NULL,
                    from_user VARCHAR(64) NOT NULL,
                    to_user VARCHAR(64) NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME(6) NOT NULL,
                    INDEX idx_expert_thread (thread_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )


async def _fetchone(sql: str, args: tuple = ()) -> Optional[dict]:
    async with _pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(sql, args)
            return await cur.fetchone()


async def _fetchall(sql: str, args: tuple = ()) -> list[dict]:
    async with _pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(sql, args)
            return await cur.fetchall()


async def _execute(sql: str, args: tuple = ()) -> int:
    async with _pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, args)
            return cur.rowcount


# ---------- Users ----------

def _user_row(row: dict) -> dict:
    row.pop("password", None)
    row["created_at"] = _utc(row.get("created_at"))
    if row.get("is_healthy") is not None:
        row["is_healthy"] = bool(row["is_healthy"])
    return row


async def get_user_by_id(user_id: str) -> Optional[dict]:
    row = await _fetchone("SELECT * FROM users WHERE user_id = %s", (user_id,))
    return _user_row(row) if row else None


async def get_user_by_email(email: str, include_password: bool = False) -> Optional[dict]:
    row = await _fetchone("SELECT * FROM users WHERE email = %s", (email.lower(),))
    if not row:
        return None
    if not include_password:
        row.pop("password", None)
    row["created_at"] = _utc(row.get("created_at"))
    return row


async def create_user(doc: dict) -> None:
    await _execute(
        """
        INSERT INTO users (user_id, email, name, phone, role, language, picture, location, password, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            doc["user_id"],
            doc["email"].lower(),
            doc["name"],
            doc.get("phone"),
            doc.get("role", "farmer"),
            doc.get("language", "en"),
            doc.get("picture"),
            doc.get("location"),
            doc.get("password"),
            doc["created_at"],
        ),
    )


async def update_user(user_id: str, fields: dict) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k} = %s" for k in fields)
    await _execute(
        f"UPDATE users SET {cols} WHERE user_id = %s",
        (*fields.values(), user_id),
    )


async def count_users(role: Optional[str] = None) -> int:
    if role:
        row = await _fetchone("SELECT COUNT(*) AS c FROM users WHERE role = %s", (role,))
    else:
        row = await _fetchone("SELECT COUNT(*) AS c FROM users")
    return int(row["c"]) if row else 0


# ---------- Sessions ----------

async def get_session(session_token: str) -> Optional[dict]:
    row = await _fetchone(
        "SELECT user_id, session_token, expires_at, created_at FROM user_sessions WHERE session_token = %s",
        (session_token,),
    )
    if row:
        row["expires_at"] = _utc(row.get("expires_at"))
        row["created_at"] = _utc(row.get("created_at"))
    return row


async def upsert_session(session_token: str, user_id: str, expires_at: datetime, created_at: datetime) -> None:
    await _execute(
        """
        INSERT INTO user_sessions (session_token, user_id, expires_at, created_at)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE user_id = VALUES(user_id), expires_at = VALUES(expires_at)
        """,
        (session_token, user_id, expires_at, created_at),
    )


# ---------- Scans ----------

def _scan_row(row: dict) -> dict:
    row["is_healthy"] = bool(row["is_healthy"])
    row["created_at"] = _utc(row.get("created_at"))
    row["next_actions"] = _json_loads(row["next_actions"]) or []
    row["treatments"] = _json_loads(row["treatments"]) or []
    row["preventive_measures"] = _json_loads(row["preventive_measures"]) or []
    return row


def _chat_row(row: dict) -> dict:
    row["timestamp"] = _utc(row.get("timestamp"))
    return row


async def insert_scan(scan: dict) -> None:
    await _execute(
        """
        INSERT INTO scans (
            scan_id, user_id, crop_name, image_base64, disease_name, is_healthy,
            severity, confidence, symptoms, causes, affected_part, spread_risk,
            next_actions, treatments, preventive_measures, notes, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            scan["scan_id"],
            scan["user_id"],
            scan.get("crop_name"),
            scan["image_base64"],
            scan["disease_name"],
            int(scan["is_healthy"]),
            scan["severity"],
            scan["confidence"],
            scan["symptoms"],
            scan["causes"],
            scan["affected_part"],
            scan["spread_risk"],
            _json_dumps(scan.get("next_actions", [])),
            _json_dumps(scan.get("treatments", [])),
            _json_dumps(scan.get("preventive_measures", [])),
            scan.get("notes"),
            scan["created_at"],
        ),
    )


async def list_scans(user_id: str, limit: int = 50) -> list[dict]:
    rows = await _fetchall(
        "SELECT * FROM scans WHERE user_id = %s ORDER BY created_at DESC LIMIT %s",
        (user_id, limit),
    )
    return [_scan_row(r) for r in rows]


async def get_scan(scan_id: str, user_id: str) -> Optional[dict]:
    row = await _fetchone(
        "SELECT * FROM scans WHERE scan_id = %s AND user_id = %s",
        (scan_id, user_id),
    )
    return _scan_row(row) if row else None


async def delete_scan(scan_id: str, user_id: str) -> int:
    return await _execute(
        "DELETE FROM scans WHERE scan_id = %s AND user_id = %s",
        (scan_id, user_id),
    )


async def list_scans_for_user(user_id: str, limit: int = 200) -> list[dict]:
    rows = await _fetchall(
        "SELECT * FROM scans WHERE user_id = %s ORDER BY created_at DESC LIMIT %s",
        (user_id, limit),
    )
    return [_scan_row(r) for r in rows]


async def count_scans(healthy_only: Optional[bool] = None) -> int:
    if healthy_only is False:
        row = await _fetchone("SELECT COUNT(*) AS c FROM scans WHERE is_healthy = 0")
    else:
        row = await _fetchone("SELECT COUNT(*) AS c FROM scans")
    return int(row["c"]) if row else 0


async def top_diseases(limit: int = 5) -> list[dict]:
    rows = await _fetchall(
        """
        SELECT disease_name AS name, COUNT(*) AS count
        FROM scans WHERE is_healthy = 0
        GROUP BY disease_name ORDER BY count DESC LIMIT %s
        """,
        (limit,),
    )
    return rows


# ---------- Chatbot ----------

async def list_chatbot_history(session_id: str, limit: int = 40) -> list[dict]:
    rows = await _fetchall(
        """
        SELECT session_id, user_id, role, content, timestamp
        FROM chatbot_msgs WHERE session_id = %s ORDER BY timestamp ASC LIMIT %s
        """,
        (session_id, limit),
    )
    return [_chat_row(r) for r in rows]


async def insert_chatbot_messages(messages: list[dict]) -> None:
    async with _pool.acquire() as conn:
        async with conn.cursor() as cur:
            for m in messages:
                await cur.execute(
                    """
                    INSERT INTO chatbot_msgs (session_id, user_id, role, content, timestamp)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (m["session_id"], m["user_id"], m["role"], m["content"], m["timestamp"]),
                )


async def list_chatbot_by_user(user_id: str, session_id: Optional[str] = None, limit: int = 200) -> list[dict]:
    if session_id:
        rows = await _fetchall(
            """
            SELECT session_id, user_id, role, content, timestamp
            FROM chatbot_msgs WHERE user_id = %s AND session_id = %s
            ORDER BY timestamp ASC LIMIT %s
            """,
            (user_id, session_id, limit),
        )
    else:
        rows = await _fetchall(
            """
            SELECT session_id, user_id, role, content, timestamp
            FROM chatbot_msgs WHERE user_id = %s ORDER BY timestamp ASC LIMIT %s
            """,
            (user_id, limit),
        )
    return [_chat_row(r) for r in rows]


# ---------- Expert messages ----------

async def insert_expert_message(msg: dict) -> None:
    await _execute(
        """
        INSERT INTO expert_messages (thread_id, from_user, to_user, content, timestamp)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (msg["thread_id"], msg["from_user"], msg["to_user"], msg["content"], msg["timestamp"]),
    )


async def list_expert_messages(thread_id: str, limit: int = 200) -> list[dict]:
    rows = await _fetchall(
        """
        SELECT thread_id, from_user, to_user, content, timestamp
        FROM expert_messages WHERE thread_id = %s ORDER BY timestamp ASC LIMIT %s
        """,
        (thread_id, limit),
    )
    return [_chat_row(r) for r in rows]
