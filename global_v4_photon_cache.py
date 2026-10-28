#!/usr/bin/env python3
"""
Build a resumable raw Photon response cache for unique event-location strings.

This stage intentionally DOES NOT choose the final Photon candidate and DOES NOT
perform the World Bank boundary spatial join. It stores the exact raw Photon
response (limit=10) for each unique raw location string. That lets the existing
project scoring logic be preserved later rather than reimplemented here.

Properties:
- exact raw location strings; no lowercasing/normalization
- frequency-prioritized queue
- SQLite WAL cache
- bounded concurrency
- retries with exponential backoff
- resumable
- safe interruption / periodic commits
- no dependency on the public photon.komoot.io endpoint by default

Input JSONL:
  {"count": 6090231, "location": "Gaza"}
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import sqlite3
import threading
import time
from pathlib import Path

import requests


_tls = threading.local()


def get_session():
    s = getattr(_tls, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({
            "User-Agent": "ZHAI-global-v4-geocoding/1.0"
        })
        _tls.session = s
    return s


def connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, timeout=60)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("PRAGMA busy_timeout=60000")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS photon_cache (
            location TEXT PRIMARY KEY,
            mention_count INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            http_status INTEGER,
            feature_count INTEGER,
            response_json TEXT,
            error TEXT,
            updated_at TEXT
        )
        """
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_photon_cache_status_count "
        "ON photon_cache(status, mention_count DESC)"
    )
    con.commit()
    return con


def load_queue(con: sqlite3.Connection, queue_path: Path, batch_size: int = 10000):
    before = con.execute("SELECT COUNT(*) FROM photon_cache").fetchone()[0]
    batch = []

    with queue_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            location = obj["location"]
            count = int(obj["count"])
            if not isinstance(location, str) or not location:
                raise SystemExit(f"Invalid location at queue line {line_no}")

            batch.append((location, count))
            if len(batch) >= batch_size:
                con.executemany(
                    "INSERT OR IGNORE INTO photon_cache(location, mention_count) "
                    "VALUES (?, ?)",
                    batch,
                )
                con.commit()
                batch.clear()

    if batch:
        con.executemany(
            "INSERT OR IGNORE INTO photon_cache(location, mention_count) VALUES (?, ?)",
            batch,
        )
        con.commit()

    after = con.execute("SELECT COUNT(*) FROM photon_cache").fetchone()[0]
    print(f"cache rows before : {before:,}")
    print(f"cache rows after  : {after:,}")
    print(f"new queue rows    : {after-before:,}")


def fetch_one(
    endpoint: str,
    location: str,
    already_attempted: int,
    timeout: float,
    max_attempts: int,
):
    last_error = None
    last_status = None
    tries_this_run = 0
    remaining = max(0, max_attempts - already_attempted)

    for local_try in range(remaining):
        tries_this_run += 1
        try:
            r = get_session().get(
                endpoint,
                params={"q": location, "limit": 10},
                timeout=timeout,
            )
            last_status = r.status_code

            if r.status_code == 200:
                obj = r.json()
                features = obj.get("features")
                if not isinstance(features, list):
                    raise ValueError("Photon response has non-list 'features'")
                return {
                    "status": "ok",
                    "http_status": 200,
                    "feature_count": len(features),
                    "response_json": json.dumps(
                        obj, ensure_ascii=False, separators=(",", ":")
                    ),
                    "error": None,
                    "tries": tries_this_run,
                }

            # Retry only transient server/throttling responses.
            if r.status_code not in (408, 425, 429, 500, 502, 503, 504):
                return {
                    "status": "failed",
                    "http_status": r.status_code,
                    "feature_count": None,
                    "response_json": None,
                    "error": f"HTTP {r.status_code}: {r.text[:500]}",
                    "tries": tries_this_run,
                }

            last_error = f"HTTP {r.status_code}: {r.text[:300]}"

        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"

        if local_try + 1 < remaining:
            # Private service should normally not need this, but it protects
            # against startup/load spikes and transient failures.
            delay = min(30.0, (2 ** local_try) + random.random())
            time.sleep(delay)

    return {
        "status": "failed",
        "http_status": last_status,
        "feature_count": None,
        "response_json": None,
        "error": last_error or "unknown failure",
        "tries": tries_this_run,
    }


def pending_iter(con, max_attempts: int, limit: int | None):
    sql = (
        "SELECT location, mention_count, attempts "
        "FROM photon_cache "
        "WHERE status != 'ok' AND attempts < ? "
        "ORDER BY mention_count DESC, location"
    )
    params = [max_attempts]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)

    cur = con.execute(sql, params)
    for row in cur:
        yield row


def print_db_stats(con):
    total, mentions = con.execute(
        "SELECT COUNT(*), COALESCE(SUM(mention_count),0) FROM photon_cache"
    ).fetchone()
    ok, ok_mentions = con.execute(
        "SELECT COUNT(*), COALESCE(SUM(mention_count),0) "
        "FROM photon_cache WHERE status='ok'"
    ).fetchone()
    empty = con.execute(
        "SELECT COUNT(*) FROM photon_cache "
        "WHERE status='ok' AND feature_count=0"
    ).fetchone()[0]
    failed = con.execute(
        "SELECT COUNT(*) FROM photon_cache WHERE status='failed'"
    ).fetchone()[0]

    pct_u = (100.0 * ok / total) if total else 0.0
    pct_m = (100.0 * ok_mentions / mentions) if mentions else 0.0

    print(f"total unique       : {total:,}")
    print(f"total mentions     : {mentions:,}")
    print(f"successful unique  : {ok:,} ({pct_u:.4f}%)")
    print(f"successful mentions: {ok_mentions:,} ({pct_m:.4f}%)")
    print(f"zero-feature       : {empty:,}")
    print(f"failed             : {failed:,}")


def export_jsonl(con, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        cur = con.execute(
            "SELECT location, mention_count, feature_count, response_json "
            "FROM photon_cache WHERE status='ok' "
            "ORDER BY mention_count DESC, location"
        )
        for location, count, feature_count, response_json in cur:
            out.write(
                json.dumps(
                    {
                        "location": location,
                        "count": count,
                        "feature_count": feature_count,
                        "photon": json.loads(response_json),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            n += 1
    print(f"exported           : {n:,}")
    print(f"export path        : {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queue_jsonl", type=Path)
    ap.add_argument("--db", type=Path, default=Path("global_v4_photon_cache.sqlite"))
    ap.add_argument(
        "--endpoint",
        default="http://127.0.0.1:2322/api",
        help="Private Photon API endpoint.",
    )
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--commit-every", type=int, default=100)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional smoke-test limit only. Omit for the full production run.",
    )
    ap.add_argument(
        "--stats-only",
        action="store_true",
        help="Load queue if needed, print cache statistics, and exit.",
    )
    ap.add_argument(
        "--export-jsonl",
        type=Path,
        default=None,
        help="After processing, export successful raw Photon responses to JSONL.",
    )
    args = ap.parse_args()

    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be >= 1")

    con = connect(args.db)
    load_queue(con, args.queue_jsonl)

    if args.stats_only:
        print_db_stats(con)
        if args.export_jsonl:
            export_jsonl(con, args.export_jsonl)
        return

    # Connectivity check before scheduling millions of requests.
    try:
        test = get_session().get(
            args.endpoint,
            params={"q": "Berlin", "limit": 1},
            timeout=args.timeout,
        )
        test.raise_for_status()
        test_obj = test.json()
        if not isinstance(test_obj.get("features"), list):
            raise ValueError("health-check response has no list-valued features")
    except Exception as e:
        raise SystemExit(
            f"Photon connectivity check failed for {args.endpoint}: {e}"
        )

    work = pending_iter(con, args.max_attempts, args.limit)
    executor = cf.ThreadPoolExecutor(max_workers=args.concurrency)

    max_inflight = max(args.concurrency * 4, 16)
    inflight = {}
    done_total = 0
    done_ok = 0
    done_failed = 0
    done_empty = 0
    mention_ok = 0
    started = time.time()
    since_commit = 0

    def submit_until_full():
        nonlocal work
        while len(inflight) < max_inflight:
            try:
                location, mention_count, attempts = next(work)
            except StopIteration:
                break
            fut = executor.submit(
                fetch_one,
                args.endpoint,
                location,
                attempts,
                args.timeout,
                args.max_attempts,
            )
            inflight[fut] = (location, mention_count, attempts)

    submit_until_full()

    try:
        while inflight:
            done, _ = cf.wait(inflight, return_when=cf.FIRST_COMPLETED)
            for fut in done:
                location, mention_count, old_attempts = inflight.pop(fut)
                res = fut.result()
                new_attempts = old_attempts + res["tries"]

                con.execute(
                    """
                    UPDATE photon_cache
                    SET status=?,
                        attempts=?,
                        http_status=?,
                        feature_count=?,
                        response_json=?,
                        error=?,
                        updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')
                    WHERE location=?
                    """,
                    (
                        res["status"],
                        new_attempts,
                        res["http_status"],
                        res["feature_count"],
                        res["response_json"],
                        res["error"],
                        location,
                    ),
                )

                done_total += 1
                since_commit += 1
                if res["status"] == "ok":
                    done_ok += 1
                    mention_ok += mention_count
                    if res["feature_count"] == 0:
                        done_empty += 1
                else:
                    done_failed += 1

                if since_commit >= args.commit_every:
                    con.commit()
                    since_commit = 0

                if done_total % 1000 == 0:
                    elapsed = max(time.time() - started, 1e-9)
                    print(
                        f"processed={done_total:,} "
                        f"ok={done_ok:,} failed={done_failed:,} "
                        f"empty={done_empty:,} "
                        f"run_qps={done_total/elapsed:.2f} "
                        f"run_mentions_covered={mention_ok:,}",
                        flush=True,
                    )

            submit_until_full()

    except KeyboardInterrupt:
        print("\nInterrupted; committing completed results before exit.", flush=True)
    finally:
        con.commit()
        executor.shutdown(wait=True, cancel_futures=False)

    print()
    print_db_stats(con)

    if args.export_jsonl:
        export_jsonl(con, args.export_jsonl)


if __name__ == "__main__":
    main()
