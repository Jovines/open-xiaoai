#!/usr/bin/env python3
"""Publish offline ASR results as an idempotent, read-only historical replay."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import urllib.error
import urllib.request


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="发布离线重转写结果，不覆盖原事件")
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--audit-id", required=True)
    parser.add_argument("--url", default=os.environ.get("ZERIS_AUDIO_INGEST_URL", ""))
    parser.add_argument("--token", default=os.environ.get("ZERIS_AUDIO_INGEST_TOKEN", ""))
    return parser.parse_args(argv)


def replay_event(event: dict, audit_id: str, replayed_at: str | None = None) -> dict:
    original_id = str(event["event_id"])
    digest = hashlib.sha256(f"{audit_id}:{original_id}".encode()).hexdigest()[:24]
    copy = json.loads(json.dumps(event))
    copy["event_id"] = f"audio-replay-{digest}"
    provenance = copy.setdefault("provenance", {})
    provenance["historical_replay"] = {
        "audit_id": audit_id,
        "original_event_id": original_id,
        "replayed_at": replayed_at or dt.datetime.now(dt.timezone.utc).isoformat(),
        "read_only_replay": True,
    }
    return copy


def post(url: str, token: str, event: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(event, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.url or not args.token:
        raise SystemExit("需要 ZERIS_AUDIO_INGEST_URL 和 ZERIS_AUDIO_INGEST_TOKEN")
    if not args.audit_id or len(args.audit_id) > 64:
        raise SystemExit("audit-id 必须是 1..64 字符")
    sent = duplicates = failed = 0
    for pending in sorted(args.evidence_dir.rglob("*.event.pending.json")):
        try:
            event = replay_event(json.loads(pending.read_text(encoding="utf-8")), args.audit_id)
            result = post(args.url, args.token, event)
            if not result.get("ok"):
                raise RuntimeError(str(result.get("error") or "Zeris rejected event"))
            duplicates += int(bool(result.get("duplicate")))
            sent += 1
            completed = Path(str(pending).replace(".event.pending.json", ".event.replay.json"))
            pending.replace(completed)
        except (OSError, ValueError, KeyError, RuntimeError, urllib.error.URLError) as error:
            failed += 1
            print(json.dumps({"file": str(pending), "error": str(error)}, ensure_ascii=False))
    print(json.dumps({"audit_id": args.audit_id, "published": sent, "duplicates": duplicates, "failed": failed}, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
