"""
Loader and parser for Telegram Chat Export JSON files.
"""

import contextlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

with contextlib.suppress(ImportError):
    pass

from src.ingestion.schema import NormalizedMessage

logger = logging.getLogger(__name__)


def extract_raw_text(raw_text: Any) -> str:
    """
    Extract a unified plain text string from Telegram's heterogeneous text format.
    Telegram exports text as either:
    1. A plain string: "Hello world"
    2. A list of strings and entity dicts: ["Hello ", {"type": "bold", "text": "world"}, ...]
    """
    if raw_text is None:
        return ""
    if isinstance(raw_text, str):
        return raw_text.strip()
    if isinstance(raw_text, list):
        parts = []
        for elem in raw_text:
            if isinstance(elem, str):
                parts.append(elem)
            elif isinstance(elem, dict) and "text" in elem:
                parts.append(str(elem["text"]))
            elif isinstance(elem, (int, float)):
                parts.append(str(elem))
        return "".join(parts).strip()
    return str(raw_text).strip()


def parse_timestamp(raw_msg: dict[str, Any]) -> tuple[datetime, int]:
    """Parse timestamp and unixtime from raw message object."""
    unixtime = raw_msg.get("date_unixtime")
    if unixtime is not None:
        try:
            ts = datetime.fromtimestamp(int(unixtime))
            return ts, int(unixtime)
        except Exception:
            pass

    date_str = raw_msg.get("date") or raw_msg.get("datetime") or raw_msg.get("timestamp")
    if date_str:
        if isinstance(date_str, (int, float)):
            return datetime.fromtimestamp(date_str), int(date_str)
        # Try ISO format
        try:
            clean_date = str(date_str).replace("Z", "+00:00")
            ts = datetime.fromisoformat(clean_date)
            return ts, int(ts.timestamp())
        except Exception:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d.%m.%Y %H:%M:%S"):
            try:
                ts = datetime.strptime(str(date_str), fmt)
                return ts, int(ts.timestamp())
            except Exception:
                continue

    # Fallback to epoch
    epoch = datetime(1970, 1, 1)
    return epoch, 0


def load_export_file(file_path: str | Path) -> tuple[dict[str, Any], list[NormalizedMessage]]:
    """
    Load a Telegram chat export result.json file and return metadata and normalized messages.
    """
    path = Path(file_path)
    if not path.is_file():
        # Check if it's a directory containing result.json
        candidate = path / "result.json"
        if candidate.is_file():
            path = candidate
        else:
            raise FileNotFoundError(f"Cannot find export file at {file_path}")

    logger.info(f"Loading export from {path}...")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    chat_info = {
        "name": data.get("name", path.parent.name),
        "type": data.get("type", "unknown"),
        "id": int(data.get("id", 0)) if data.get("id") else 0,
        "file_path": str(path),
    }

    raw_msgs = data.get("messages", [])
    if not raw_msgs and isinstance(data, list):
        raw_msgs = data

    normalized_msgs: list[NormalizedMessage] = []
    chat_id = chat_info["id"]
    chat_name = chat_info["name"]

    for raw in raw_msgs:
        if not isinstance(raw, dict):
            continue

        msg_id = raw.get("id", 0)
        msg_type = raw.get("type", "message")
        is_service = msg_type == "service" or "action" in raw

        author = (
            raw.get("from")
            or raw.get("actor")
            or raw.get("from_id")
            or raw.get("actor_id")
            or raw.get("sender")
            or "Anonymous"
        )
        author_id = str(raw.get("from_id") or raw.get("actor_id") or raw.get("user_id") or author)

        text = extract_raw_text(raw.get("text"))

        # Check for media captions
        has_media = bool(
            raw.get("photo")
            or raw.get("file")
            or raw.get("media_type")
            or raw.get("voice_message")
            or raw.get("video_file")
        )
        media_type = raw.get("media_type")
        if not media_type:
            if raw.get("photo"):
                media_type = "photo"
            elif raw.get("voice_message"):
                media_type = "voice"
            elif raw.get("video_file"):
                media_type = "video"
            elif raw.get("file"):
                media_type = "document"

        reply_to_id = raw.get("reply_to_message_id")
        if reply_to_id is not None:
            try:
                reply_to_id = int(reply_to_id)
            except (ValueError, TypeError):
                reply_to_id = None

        forwarded_from = raw.get("forwarded_from")

        ts, unixtime = parse_timestamp(raw)

        norm_msg = NormalizedMessage(
            msg_id=int(msg_id) if msg_id else len(normalized_msgs) + 1,
            chat_id=chat_id,
            chat_name=chat_name,
            timestamp=ts,
            unixtime=unixtime,
            author_raw=str(author),
            author_id_raw=str(author_id),
            text_raw=text,
            reply_to_id=reply_to_id,
            has_media=has_media,
            media_type=media_type,
            is_service=is_service,
            forwarded_from=str(forwarded_from) if forwarded_from else None,
        )
        normalized_msgs.append(norm_msg)

    logger.info(f"Loaded {len(normalized_msgs)} messages from {chat_name}")
    return chat_info, normalized_msgs


def merge_multiple_exports(export_dirs: list[str | Path]) -> tuple[list[dict[str, Any]], list[NormalizedMessage]]:
    """
    Load and merge multiple Telegram export directories into a single unified stream.
    Sorts all messages chronologically.
    """
    all_chats_info: list[dict[str, Any]] = []
    all_messages: list[NormalizedMessage] = []

    for d in export_dirs:
        try:
            info, msgs = load_export_file(d)
            all_chats_info.append(info)
            all_messages.extend(msgs)
        except Exception as e:
            logger.error(f"Failed to load export from {d}: {e}")

    # Sort all messages chronologically by timestamp / unixtime
    all_messages.sort(key=lambda m: (m.unixtime, m.msg_id))
    logger.info(f"Total merged messages across {len(all_chats_info)} chats: {len(all_messages)}")
    return all_chats_info, all_messages
