#!/usr/bin/env python3

import argparse
import csv
import getpass
import io
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json


REPO = Path("/mnt/c/Users/Bonaventure/Downloads/event-location-extraction")
FINAL_DIR = REPO / "global_v4_audit" / "final_geocoded"
META_ROOT = Path("/home/bonadossou/global_v4_work/compact_event_metadata_v1")
AUDIT_DIR = REPO / "global_v4_audit" / "db_ingest_v4"

RUN_ID = "global_v4_qwen3_5_4b_20260903"
ONTOLOGY_VERSION = "bona.v4"
MODEL_NAME = "qwen3.5-4b-v4-a100-40gb-20260903-221617"

EXPECTED_SHARDS = 200
EXPECTED_EVENT_ARTICLES = 29_236_324
EXPECTED_EVENTS = 128_267_182
EXPECTED_LOCATION_COMPONENTS = 137_216_336

NULL = "__ZHAI_V4_NULL_6A8C2C9A__"

EVENT_KNOWN_KEYS = {
    "event_type",
    "grounding_quote",
    "event_location",
    "event_location_admin_level",
    "event_time",
    "time_status",
    "severity",
    "geotaxonomy",
    "event_location_raw",
    "event_location_admin_level_raw",
}

GEO_KNOWN_KEYS = {
    "component_index",
    "query",
    "raw_admin_level",
    "photon_layer",
    "photon_resolution_status",
    "photon_selected_query",
    "photon_selected_layer",
    "photon_fallback_step",
    "photon_fallback_label",
    "resolved_name",
    "zhai",
    "photon_type",
    "lat",
    "lon",
    "osm_id",
    "osm_type",
    "country",
    "countrycode",
    "wb_level",
    "wb_match_status",
    "adm_code",
}

EVENT_COLUMNS = [
    "run_id",
    "publish_month",
    "article_published_at",
    "shard",
    "event_key",
    "article_uri",
    "original_event_index",
    "language",
    "event_type",
    "grounding_quote",
    "event_location",
    "event_location_admin_level",
    "event_time",
    "time_status",
    "severity",
    "event_location_raw",
    "event_location_admin_level_raw",
    "raw_event_time",
    "temporal_syntax_kind",
    "temporal_parse_status",
    "temporal_review_reason",
    "temporal_start_token",
    "temporal_end_token",
    "temporal_start_precision",
    "temporal_end_precision",
    "temporal_lower_bound_date",
    "temporal_upper_bound_date",
    "extra",
]

LOCATION_COLUMNS = [
    "run_id",
    "publish_month",
    "shard",
    "event_key",
    "article_uri",
    "component_index",
    "query",
    "raw_admin_level",
    "photon_layer",
    "photon_resolution_status",
    "photon_selected_query",
    "photon_selected_layer",
    "photon_fallback_step",
    "photon_fallback_label",
    "resolved_name",
    "zhai",
    "photon_type",
    "lat",
    "lon",
    "osm_id",
    "osm_type",
    "country",
    "countrycode",
    "wb_level",
    "wb_match_status",
    "adm_code",
    "extra",
]


def nullish(v):
    if v is None or v == "":
        return None
    return v


def extra_json(obj, known):
    extra = {k: v for k, v in obj.items() if k not in known}
    if not extra:
        return None
    return json.dumps(extra, ensure_ascii=False, separators=(",", ":"))


def normalize_cell(v):
    if v is None:
        return NULL
    if isinstance(v, bool):
        s = "true" if v else "false"
    else:
        s = str(v)
    if s == NULL:
        raise ValueError(f"Input value collides with COPY NULL sentinel: {NULL}")
    return s


def csv_chunks(rows, chunk_chars=1 << 20):
    buf = io.StringIO()
    writer = csv.writer(
        buf,
        delimiter="\t",
        quotechar='"',
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )

    for row in rows:
        writer.writerow([normalize_cell(v) for v in row])

        if buf.tell() >= chunk_chars:
            yield buf.getvalue()
            buf.seek(0)
            buf.truncate(0)

    if buf.tell():
        yield buf.getvalue()


class IteratorReader:
    def __init__(self, chunks):
        self._chunks = iter(chunks)
        self._buffer = ""

    def read(self, size=-1):
        if size is None or size < 0:
            parts = [self._buffer]
            self._buffer = ""
            parts.extend(self._chunks)
            return "".join(parts)

        while len(self._buffer) < size:
            try:
                self._buffer += next(self._chunks)
            except StopIteration:
                break

        out = self._buffer[:size]
        self._buffer = self._buffer[size:]
        return out


class Connector:
    def __init__(self, host, port, dbname, user):
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = os.environ.get("PGPASSWORD")

    def connect(self):
        kwargs = {
            "host": self.host,
            "port": self.port,
            "dbname": self.dbname,
            "user": self.user,
            "application_name": "global_v4_ingestion",
            "connect_timeout": 30,
        }

        if self.password:
            kwargs["password"] = self.password

        try:
            return psycopg2.connect(**kwargs)
        except psycopg2.OperationalError as exc:
            msg = str(exc).lower()

            if self.password is None and (
                "password" in msg or "authentication" in msg
            ):
                self.password = getpass.getpass(
                    f"PostgreSQL password for {self.user}: "
                )
                kwargs["password"] = self.password
                return psycopg2.connect(**kwargs)

            raise


def paths_for_shard(shard):
    s = f"{shard:03d}"

    article_path = META_ROOT / "articles" / f"shard-{s}.parquet"
    event_path = META_ROOT / "events" / f"shard-{s}.parquet"
    geo_path = FINAL_DIR / f"shard-{s}.geo.jsonl"

    for p in (article_path, event_path, geo_path):
        if not p.exists():
            raise FileNotFoundError(p)

    return article_path, event_path, geo_path


def publish_month_from_timestamp(value):
    s = str(value).strip()

    if (
        len(s) < 7
        or s[4] != "-"
        or s[7:8] not in ("", "-", " ", "T")
    ):
        raise ValueError(f"Unexpected publish_date: {value!r}")

    year = int(s[0:4])
    month = int(s[5:7])

    if not 1 <= month <= 12:
        raise ValueError(f"Unexpected publish month: {value!r}")

    return f"{year:04d}-{month:02d}-01"


def load_article_metadata(shard):
    article_path, _, _ = paths_for_shard(shard)

    table = pq.read_table(
        article_path,
        columns=["id", "language", "publish_date"],
    )

    ids = table["id"].to_pylist()
    languages = table["language"].to_pylist()
    dates = table["publish_date"].to_pylist()

    out = {}

    for article_id, language, published_at in zip(ids, languages, dates):
        if article_id in out:
            raise RuntimeError(
                f"duplicate article metadata id in shard {shard}: {article_id}"
            )

        if not published_at:
            raise RuntimeError(
                f"missing publish_date in shard {shard}: {article_id}"
            )

        out[article_id] = (
            language,
            str(published_at),
            publish_month_from_timestamp(published_at),
        )

    return out


def load_event_metadata(shard, article_meta):
    _, event_path, _ = paths_for_shard(shard)

    columns = [
        "event_key",
        "id",
        "event_index",
        "raw_event_time",
        "syntax_kind",
        "parse_status",
        "review_reason",
        "start_token",
        "end_token",
        "start_precision",
        "end_precision",
        "lower_bound_date",
        "upper_bound_date",
    ]

    pf = pq.ParquetFile(event_path)

    by_article = defaultdict(list)
    total = 0

    for batch in pf.iter_batches(batch_size=100_000, columns=columns):
        vals = [batch.column(i).to_pylist() for i in range(len(columns))]

        for row in zip(*vals):
            (
                event_key,
                article_id,
                event_index,
                raw_event_time,
                syntax_kind,
                parse_status,
                review_reason,
                start_token,
                end_token,
                start_precision,
                end_precision,
                lower_bound_date,
                upper_bound_date,
            ) = row

            if article_id not in article_meta:
                raise RuntimeError(
                    f"event metadata article missing from article parquet: "
                    f"shard={shard} article={article_id}"
                )

            if not isinstance(event_key, (bytes, bytearray)):
                raise RuntimeError(
                    f"non-binary event_key: shard={shard} article={article_id}"
                )

            if len(event_key) != 32:
                raise RuntimeError(
                    f"event_key length != 32: shard={shard} article={article_id}"
                )

            by_article[article_id].append(
                (
                    bytes(event_key),
                    int(event_index),
                    raw_event_time,
                    syntax_kind,
                    parse_status,
                    review_reason,
                    start_token,
                    end_token,
                    start_precision,
                    end_precision,
                    lower_bound_date,
                    upper_bound_date,
                )
            )

            total += 1

    # Preserve the original event index semantics while ensuring that the
    # surviving metadata rows are in original event order.
    for article_id, rows in by_article.items():
        rows.sort(key=lambda r: r[1])

        for i in range(1, len(rows)):
            if rows[i - 1][1] >= rows[i][1]:
                raise RuntimeError(
                    f"duplicate/decreasing original event index: "
                    f"shard={shard} article={article_id}"
                )

    if set(article_meta) != set(by_article):
        only_articles = set(article_meta) - set(by_article)
        only_events = set(by_article) - set(article_meta)

        raise RuntimeError(
            f"article/event parquet article-set mismatch for shard {shard}: "
            f"article_only={len(only_articles):,} "
            f"event_only={len(only_events):,}"
        )

    return dict(by_article), total


def iter_final_articles(geo_path):
    with geo_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise RuntimeError(
                    f"JSON error {geo_path}:{lineno}: {exc}"
                ) from exc

            if "id" not in obj:
                raise RuntimeError(
                    f"missing article id {geo_path}:{lineno}"
                )

            yield obj


def validate_alignment(shard, article_meta, event_meta, expected_event_count):
    _, _, geo_path = paths_for_shard(shard)

    seen_event_articles = set()

    total_articles = 0
    event_articles = 0
    events = 0
    locations = 0

    for obj in iter_final_articles(geo_path):
        total_articles += 1
        article_id = obj["id"]
        preds = obj.get("predictions") or []

        if not preds:
            if article_id in event_meta:
                raise RuntimeError(
                    f"metadata exists but final predictions empty: "
                    f"shard={shard} article={article_id}"
                )
            continue

        if article_id in seen_event_articles:
            raise RuntimeError(
                f"duplicate event-bearing article in final JSONL: "
                f"shard={shard} article={article_id}"
            )

        seen_event_articles.add(article_id)
        event_articles += 1

        metas = event_meta.get(article_id)

        if metas is None:
            raise RuntimeError(
                f"final event article missing metadata: "
                f"shard={shard} article={article_id}"
            )

        if len(preds) != len(metas):
            raise RuntimeError(
                f"event count mismatch: shard={shard} article={article_id} "
                f"final={len(preds)} metadata={len(metas)}"
            )

        for pred, meta in zip(preds, metas):
            raw_event_time = meta[2]

            if pred.get("event_time") != raw_event_time:
                raise RuntimeError(
                    f"event_time alignment mismatch: "
                    f"shard={shard} article={article_id} "
                    f"final={pred.get('event_time')!r} "
                    f"metadata={raw_event_time!r}"
                )

            events += 1

            geos = pred.get("geotaxonomy") or []
            seen_components = set()

            for geo in geos:
                ci = geo.get("component_index")

                if ci is None:
                    raise RuntimeError(
                        f"missing component_index: "
                        f"shard={shard} article={article_id}"
                    )

                ci = int(ci)

                if ci in seen_components:
                    raise RuntimeError(
                        f"duplicate component_index={ci}: "
                        f"shard={shard} article={article_id}"
                    )

                seen_components.add(ci)
                locations += 1

    if events != expected_event_count:
        raise RuntimeError(
            f"shard event total mismatch: "
            f"shard={shard} final={events:,} "
            f"metadata={expected_event_count:,}"
        )

    if seen_event_articles != set(event_meta):
        missing = set(event_meta) - seen_event_articles
        raise RuntimeError(
            f"{len(missing):,} metadata articles were not encountered "
            f"in final JSONL for shard {shard}"
        )

    return {
        "all_final_articles": total_articles,
        "event_articles": event_articles,
        "events": events,
        "location_components": locations,
    }


def event_rows(shard, article_meta, event_meta, expected_event_count, stats):
    _, _, geo_path = paths_for_shard(shard)

    seen = set()

    for obj in iter_final_articles(geo_path):
        article_id = obj["id"]
        preds = obj.get("predictions") or []

        if not preds:
            continue

        if article_id in seen:
            raise RuntimeError(
                f"duplicate event article during COPY: "
                f"shard={shard} article={article_id}"
            )
        seen.add(article_id)

        metas = event_meta.get(article_id)

        if metas is None or len(metas) != len(preds):
            raise RuntimeError(
                f"alignment failure during event COPY: "
                f"shard={shard} article={article_id}"
            )

        language, published_at, publish_month = article_meta[article_id]

        for pred, meta in zip(preds, metas):
            (
                event_key,
                original_event_index,
                raw_event_time,
                syntax_kind,
                parse_status,
                review_reason,
                start_token,
                end_token,
                start_precision,
                end_precision,
                lower_bound_date,
                upper_bound_date,
            ) = meta

            if pred.get("event_time") != raw_event_time:
                raise RuntimeError(
                    f"event_time mismatch during COPY: "
                    f"shard={shard} article={article_id}"
                )

            stats["events"] += 1

            yield (
                RUN_ID,
                publish_month,
                published_at,
                shard,
                "\\x" + event_key.hex(),
                article_id,
                original_event_index,
                language,
                pred.get("event_type"),
                pred.get("grounding_quote"),
                pred.get("event_location"),
                pred.get("event_location_admin_level"),
                pred.get("event_time"),
                pred.get("time_status"),
                pred.get("severity"),
                pred.get("event_location_raw"),
                pred.get("event_location_admin_level_raw"),
                nullish(raw_event_time),
                nullish(syntax_kind),
                nullish(parse_status),
                nullish(review_reason),
                nullish(start_token),
                nullish(end_token),
                nullish(start_precision),
                nullish(end_precision),
                nullish(lower_bound_date),
                nullish(upper_bound_date),
                extra_json(pred, EVENT_KNOWN_KEYS),
            )

    if stats["events"] != expected_event_count:
        raise RuntimeError(
            f"event COPY generated {stats['events']:,}; "
            f"expected {expected_event_count:,}"
        )

    if seen != set(event_meta):
        raise RuntimeError(
            f"event COPY article coverage mismatch for shard {shard}"
        )

    stats["event_articles"] = len(seen)


def location_rows(shard, article_meta, event_meta, stats):
    _, _, geo_path = paths_for_shard(shard)

    stats["locations"] = 0

    for obj in iter_final_articles(geo_path):
        article_id = obj["id"]
        preds = obj.get("predictions") or []

        if not preds:
            continue

        metas = event_meta.get(article_id)

        if metas is None or len(metas) != len(preds):
            raise RuntimeError(
                f"alignment failure during location COPY: "
                f"shard={shard} article={article_id}"
            )

        publish_month = article_meta[article_id][2]

        for pred, meta in zip(preds, metas):
            event_key = meta[0]

            if pred.get("event_time") != meta[2]:
                raise RuntimeError(
                    f"event_time mismatch during location COPY: "
                    f"shard={shard} article={article_id}"
                )

            seen_components = set()

            for geo in pred.get("geotaxonomy") or []:
                ci = geo.get("component_index")

                if ci is None:
                    raise RuntimeError(
                        f"missing component_index during COPY: "
                        f"shard={shard} article={article_id}"
                    )

                ci = int(ci)

                if ci in seen_components:
                    raise RuntimeError(
                        f"duplicate component_index={ci}: "
                        f"shard={shard} article={article_id}"
                    )

                seen_components.add(ci)
                stats["locations"] += 1

                yield (
                    RUN_ID,
                    publish_month,
                    shard,
                    "\\x" + event_key.hex(),
                    article_id,
                    ci,
                    geo.get("query"),
                    geo.get("raw_admin_level"),
                    geo.get("photon_layer"),
                    geo.get("photon_resolution_status"),
                    geo.get("photon_selected_query"),
                    geo.get("photon_selected_layer"),
                    geo.get("photon_fallback_step"),
                    geo.get("photon_fallback_label"),
                    geo.get("resolved_name"),
                    geo.get("zhai"),
                    geo.get("photon_type"),
                    geo.get("lat"),
                    geo.get("lon"),
                    geo.get("osm_id"),
                    geo.get("osm_type"),
                    geo.get("country"),
                    geo.get("countrycode"),
                    geo.get("wb_level"),
                    geo.get("wb_match_status"),
                    geo.get("adm_code"),
                    extra_json(geo, GEO_KNOWN_KEYS),
                )


def copy_rows(cur, table_name, columns, rows):
    query = sql.SQL(
        """
        COPY {} ({})
        FROM STDIN
        WITH (
            FORMAT CSV,
            DELIMITER E'\\t',
            QUOTE '"',
            ESCAPE '"',
            NULL {}
        )
        """
    ).format(
        sql.Identifier("article_event_extraction", table_name),
        sql.SQL(", ").join(map(sql.Identifier, columns)),
        sql.Literal(NULL),
    )

    reader = IteratorReader(csv_chunks(rows))
    cur.copy_expert(query.as_string(cur), reader, size=1 << 20)


def register_run(conn):
    metadata = {
        "final_geocoded_dir": str(FINAL_DIR),
        "compact_metadata_root": str(META_ROOT),
        "article_candidate_total": 34_388_866,
        "usable_location_event_total": 120_001_343,
        "location_component_total": EXPECTED_LOCATION_COMPONENTS,
    }

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO article_event_extraction.event_runs (
                    run_id,
                    ontology_version,
                    model_name,
                    expected_shards,
                    expected_event_articles,
                    expected_events,
                    expected_location_components,
                    metadata
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (run_id) DO NOTHING
                """,
                (
                    RUN_ID,
                    ONTOLOGY_VERSION,
                    MODEL_NAME,
                    EXPECTED_SHARDS,
                    EXPECTED_EVENT_ARTICLES,
                    EXPECTED_EVENTS,
                    EXPECTED_LOCATION_COMPONENTS,
                    Json(metadata),
                ),
            )

            cur.execute(
                """
                SELECT
                    ontology_version,
                    model_name,
                    expected_shards,
                    expected_event_articles,
                    expected_events,
                    expected_location_components
                FROM article_event_extraction.event_runs
                WHERE run_id = %s
                """,
                (RUN_ID,),
            )

            row = cur.fetchone()

            expected = (
                ONTOLOGY_VERSION,
                MODEL_NAME,
                EXPECTED_SHARDS,
                EXPECTED_EVENT_ARTICLES,
                EXPECTED_EVENTS,
                EXPECTED_LOCATION_COMPONENTS,
            )

            if row != expected:
                raise RuntimeError(
                    f"existing event_runs row differs from expected:\n"
                    f"db={row}\nexpected={expected}"
                )


def completed_shards(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT shard
            FROM article_event_extraction.event_ingest_shards
            WHERE run_id = %s
            ORDER BY shard
            """,
            (RUN_ID,),
        )
        return {r[0] for r in cur.fetchall()}


def write_audit(shard, payload):
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    path = AUDIT_DIR / f"shard-{shard:03d}.json"

    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def load_shard(shard, connector, dry_run=False):
    started = time.time()

    article_path, event_path, geo_path = paths_for_shard(shard)

    print(f"\n=== SHARD {shard:03d} ===", flush=True)
    print(f"articles : {article_path}", flush=True)
    print(f"events   : {event_path}", flush=True)
    print(f"geo      : {geo_path}", flush=True)

    print("loading compact article metadata...", flush=True)
    article_meta = load_article_metadata(shard)

    print("loading compact event metadata...", flush=True)
    event_meta, expected_events = load_event_metadata(
        shard, article_meta
    )

    print(
        f"metadata: {len(article_meta):,} event articles, "
        f"{expected_events:,} events",
        flush=True,
    )

    if dry_run:
        print("validating complete shard alignment...", flush=True)

        validation = validate_alignment(
            shard,
            article_meta,
            event_meta,
            expected_events,
        )

        elapsed = time.time() - started

        payload = {
            "run_id": RUN_ID,
            "shard": shard,
            "mode": "dry_run",
            **validation,
            "elapsed_seconds": elapsed,
        }

        print(json.dumps(payload, indent=2), flush=True)
        return payload

    conn = connector.connect()

    try:
        register_run(conn)

        with conn:
            with conn.cursor() as cur:
                # Prevent simultaneous loaders from ingesting the same shard.
                cur.execute(
                    """
                    SELECT pg_advisory_xact_lock(
                        hashtextextended(%s, 0)
                    )
                    """,
                    (f"{RUN_ID}:{shard}",),
                )

                cur.execute(
                    """
                    SELECT
                        article_rows,
                        event_rows,
                        location_rows
                    FROM article_event_extraction.event_ingest_shards
                    WHERE run_id = %s
                      AND shard = %s
                    """,
                    (RUN_ID, shard),
                )

                existing = cur.fetchone()

                if existing is not None:
                    print(
                        f"shard {shard:03d} already committed: "
                        f"articles={existing[0]:,} "
                        f"events={existing[1]:,} "
                        f"locations={existing[2]:,}; skipping",
                        flush=True,
                    )
                    return {
                        "run_id": RUN_ID,
                        "shard": shard,
                        "mode": "skipped",
                    }

                cur.execute("SET LOCAL statement_timeout = 0")

                stats = {
                    "events": 0,
                    "event_articles": 0,
                    "locations": 0,
                }

                print("COPY events...", flush=True)

                copy_rows(
                    cur,
                    "events",
                    EVENT_COLUMNS,
                    event_rows(
                        shard,
                        article_meta,
                        event_meta,
                        expected_events,
                        stats,
                    ),
                )

                if stats["events"] != expected_events:
                    raise RuntimeError(
                        f"event COPY count mismatch: "
                        f"{stats['events']:,} != {expected_events:,}"
                    )

                print(
                    f"COPY events complete: {stats['events']:,}",
                    flush=True,
                )

                print("COPY event_locations...", flush=True)

                copy_rows(
                    cur,
                    "event_locations",
                    LOCATION_COLUMNS,
                    location_rows(
                        shard,
                        article_meta,
                        event_meta,
                        stats,
                    ),
                )

                print(
                    f"COPY event_locations complete: "
                    f"{stats['locations']:,}",
                    flush=True,
                )

                cur.execute(
                    """
                    INSERT INTO
                        article_event_extraction.event_ingest_shards (
                            run_id,
                            shard,
                            article_rows,
                            event_rows,
                            location_rows
                        )
                    VALUES (%s,%s,%s,%s,%s)
                    """,
                    (
                        RUN_ID,
                        shard,
                        len(article_meta),
                        stats["events"],
                        stats["locations"],
                    ),
                )

        elapsed = time.time() - started

        payload = {
            "run_id": RUN_ID,
            "shard": shard,
            "mode": "committed",
            "event_articles": len(article_meta),
            "events": stats["events"],
            "location_components": stats["locations"],
            "elapsed_seconds": elapsed,
        }

        write_audit(shard, payload)

        print(
            f"COMMITTED shard {shard:03d} in "
            f"{elapsed / 60:.2f} minutes",
            flush=True,
        )

        return payload

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--shard", type=int)
    group.add_argument("--all", action="store_true")

    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5439)
    parser.add_argument("--dbname", default="zhai_db")
    parser.add_argument("--user", default="zhai")

    args = parser.parse_args()

    if args.shard is not None and not 0 <= args.shard < EXPECTED_SHARDS:
        parser.error("--shard must be 0..199")

    connector = Connector(
        args.host,
        args.port,
        args.dbname,
        args.user,
    )

    if args.shard is not None:
        load_shard(
            args.shard,
            connector,
            dry_run=args.dry_run,
        )
        return

    if args.dry_run:
        shards = range(EXPECTED_SHARDS)
    else:
        conn = connector.connect()
        try:
            register_run(conn)
            done = completed_shards(conn)
        finally:
            conn.close()

        shards = [s for s in range(EXPECTED_SHARDS) if s not in done]

        print(
            f"already committed : {len(done)}",
            flush=True,
        )
        print(
            f"remaining shards   : {len(shards)}",
            flush=True,
        )

    global_started = time.time()

    for n, shard in enumerate(shards, 1):
        print(
            f"\nGLOBAL PROGRESS {n}/{len(shards)} "
            f"(shard {shard:03d})",
            flush=True,
        )

        load_shard(
            shard,
            connector,
            dry_run=args.dry_run,
        )

    elapsed = time.time() - global_started

    print(
        f"\nALL REQUESTED SHARDS FINISHED "
        f"in {elapsed / 3600:.2f} hours",
        flush=True,
    )


if __name__ == "__main__":
    main()
