"""
Storage — Kullanıcı verisi yönetimi (bellek tabanlı)
Web uygulaması için basit in-memory store.
"""

import threading

_lock  = threading.Lock()
_users: dict = {}   # chat_id → user dict


def set_user(chat_id: int, data: dict) -> None:
    with _lock:
        _users[chat_id] = data


def get_user(chat_id: int) -> dict | None:
    with _lock:
        return _users.get(chat_id)


def update_user(chat_id: int, **kwargs) -> None:
    with _lock:
        if chat_id in _users:
            _users[chat_id].update(kwargs)


def deactivate_user(chat_id: int) -> None:
    with _lock:
        if chat_id in _users:
            _users[chat_id]["active"] = False


def get_all_active_users() -> list[tuple]:
    with _lock:
        return [
            (cid, data.copy())
            for cid, data in _users.items()
            if data.get("active", False)
        ]


def record_result(chat_id: int, win: bool) -> None:
    with _lock:
        if chat_id not in _users:
            return
        if win:
            _users[chat_id]["wins"]   = _users[chat_id].get("wins",   0) + 1
        else:
            _users[chat_id]["losses"] = _users[chat_id].get("losses", 0) + 1
