"""Disk cache — atomic writes, LRU pruning, per-model keys.

Cache is keyed per-MODEL (not per-provider): a ZenMux failover after an
OpenRouter miss reuses the same answer, saving cost.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

CACHE_DIR = Path.home() / ".claude" / "state" / "cheap-llm-cache"
CACHE_MAX_ENTRIES = 2000


def _cache_key(
    model: str,
    system: str,
    prompt: str,
    schema: tuple[str, ...] | None,
    max_output_tokens: int = 1024,
) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(b"\0")
    h.update(system.encode())
    h.update(b"\0")
    h.update(prompt.encode())
    h.update(b"\0")
    if schema:
        h.update("|".join(schema).encode())
    if max_output_tokens != 1024:
        # Preserve the pre-1.2 cache namespace for the backward-compatible
        # default; only explicitly different budgets need a new namespace.
        h.update(b"\0")
        h.update(str(max_output_tokens).encode())
    return h.hexdigest()


def _cache_get(key: str) -> dict | None:
    p = CACHE_DIR / f"{key}.json"
    if p.exists():
        try:
            value = json.loads(p.read_text())
        except Exception:
            return None
        # Shape guard: a corrupted/foreign cache file that parses as JSON but
        # isn't {"text": str} would raise KeyError/TypeError inside
        # _try_cache_hit and crash the whole cascade. Treat it as a miss.
        if isinstance(value, dict) and isinstance(value.get("text"), str):
            return value
    return None


def _cache_put(key: str, value: dict) -> None:
    # Atomic write (temp + rename) so a mid-write crash never leaves a partial
    # cache file that the next _cache_get would try to parse. Cache writes are
    # best-effort: failure here MUST NOT propagate and break a successful
    # cascade — caller already returned the value to the user.
    tmp: Path | None = None
    fd = -1
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Developer prompts and model responses can contain private project
        # context even after secret scrubbing. Repair permissive directories
        # best-effort; unusual filesystems must not break completion.
        try:
            CACHE_DIR.chmod(0o700)
        except OSError:
            pass
        # prune oldest beyond CACHE_MAX_ENTRIES
        files: list[tuple[float, Path]] = []
        for path in CACHE_DIR.glob("*.json"):
            try:
                files.append((path.stat().st_mtime, path))
            except OSError:
                # Another process may have pruned the entry after globbing.
                continue
        files.sort(key=lambda item: item[0])
        while len(files) >= CACHE_MAX_ENTRIES:
            _, oldest = files.pop(0)
            try:
                oldest.unlink(missing_ok=True)
            except OSError:
                pass
        target = CACHE_DIR / f"{key}.json"
        # Unique same-directory temp files prevent concurrent writers from
        # replacing/removing one another's temp path. mkstemp starts at 0o600;
        # os.replace is atomic because source and target share a filesystem.
        fd, tmp_name = tempfile.mkstemp(prefix=f".{key}.", suffix=".tmp", dir=CACHE_DIR)
        tmp = Path(tmp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1  # ownership transferred to the file object
            json.dump(value, handle)
        os.replace(tmp, target)
        tmp = None
        try:
            target.chmod(0o600)
        except OSError:
            pass
    except Exception:
        pass  # cache is advisory; never break the cascade on a write error
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
