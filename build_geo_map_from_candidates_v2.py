#!/usr/bin/env python3
"""
Build an authoritative article URI -> country map from the project database.

Input:
  A local CSV file or directory containing Spark CSV part files with ONE URI
  per row and no header. These should be the unique event-candidate URIs
  produced by prepare_geo_candidates_dataproc.py.

Output:
  CSV with header: uri,country

The DB lookup follows the existing project path:
article_concept_association
 -> geo_taxonomy_concept_uris_direct_match
 -> concept_uris (concept_type='loc')
 -> geo_taxonomy
and maps each matched geo concept to its ADM0 country. The current DB stores
article/concept Wikipedia URIs with http:// while the geo direct-match table
stores https://, so the join normalizes http -> https before matching.

The MATERIALIZED CTE is intentionally preserved from sample_from_db.py so
Postgres probes article_concept_association by article_uri instead of flattening
into an expensive scan.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import psycopg2

try:
    from dotenv import dotenv_values
except ImportError:
    dotenv_values = None

_TLS = threading.local()

GEO_PROBE_SQL = """
WITH assoc_rows AS MATERIALIZED (
    SELECT article_uri, concept_uri
    FROM article_concept_association
    WHERE article_uri = ANY(%(uris)s)
)
SELECT DISTINCT ar.article_uri::text, geo.adm0_code::text
FROM assoc_rows ar
JOIN concept_uris concepts
    ON concepts.concept_uri = ar.concept_uri
   AND concepts.concept_type = 'loc'
JOIN geo_taxonomy_concept_uris_direct_match geo_concepts
    ON geo_concepts.uri = regexp_replace(ar.concept_uri, '^http://', 'https://')
JOIN geo_taxonomy geo
    ON geo.adm_code = geo_concepts.adm_code
"""

COUNTRY_SQL = """
SELECT adm0_code::text, adm_name::text
FROM geo_taxonomy
WHERE adm_level = 0
"""


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="Local CSV file or directory containing Spark part-*.csv files.")
    ap.add_argument("--output", default="candidate_geo_map.csv")
    ap.add_argument("--env-file", default=".env")
    ap.add_argument("--host", default=None,
                    help="Override SQL host. For the SSH tunnel use localhost.")
    ap.add_argument("--port", type=int, default=None,
                    help="Override SQL port. For the SSH tunnel use 5439.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=1200)
    return ap.parse_args()


def load_env(path: str) -> dict:
    vals = {}
    if dotenv_values is not None and Path(path).exists():
        vals.update({k: v for k, v in dotenv_values(path).items() if v is not None})
    vals.update(os.environ)
    return vals


def connection_params(args, env):
    host = args.host or env.get("SQL_HOST") or env.get("PGHOST") or "localhost"
    port = args.port or int(env.get("SQL_PORT") or env.get("PGPORT") or 5432)
    dbname = env.get("SQL_DATABASE") or env.get("PGDATABASE")
    user = env.get("SQL_USERNAME") or env.get("PGUSER")
    password = env.get("SQL_PASSWORD") or env.get("PGPASSWORD")
    missing = [k for k, v in {"database": dbname, "user": user, "password": password}.items() if not v]
    if missing:
        raise SystemExit(
            "Missing DB settings: " + ", ".join(missing) +
            ". Provide them via the existing .env/environment; do not put secrets on the command line."
        )
    return dict(host=host, port=port, dbname=dbname, user=user, password=password)


def get_conn(params):
    conn = getattr(_TLS, "conn", None)
    if conn is None or conn.closed:
        conn = psycopg2.connect(**params)
        conn.autocommit = True
        _TLS.conn = conn
    return conn


def probe_chunk(params, uris):
    conn = get_conn(params)
    try:
        with conn.cursor() as cur:
            cur.execute(GEO_PROBE_SQL, {"uris": uris})
            return cur.fetchall()
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        _TLS.conn = None
        conn = get_conn(params)
        with conn.cursor() as cur:
            cur.execute(GEO_PROBE_SQL, {"uris": uris})
            return cur.fetchall()


def input_files(path: Path):
    if path.is_file():
        return [path]
    files = sorted(p for p in path.rglob("*.csv") if p.is_file() and not p.name.startswith("_"))
    if not files:
        raise SystemExit(f"No CSV files found under {path}")
    return files


def iter_uris(files):
    for f in files:
        with f.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.reader(fh):
                if not row:
                    continue
                uri = row[0].strip()
                if uri:
                    yield uri


def chunks(iterator, n):
    buf = []
    for item in iterator:
        buf.append(item)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def main():
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.chunk_size < 1:
        raise SystemExit("--chunk-size must be >= 1")

    env = load_env(args.env_file)
    params = connection_params(args, env)

    # Load ADM0 code -> canonical country name once.
    with psycopg2.connect(**params) as conn:
        with conn.cursor() as cur:
            cur.execute(COUNTRY_SQL)
            country_by_code = {str(code): str(name) for code, name in cur.fetchall()}

    files = input_files(Path(args.input))
    print(f"[input] {len(files)} CSV part file(s)")
    print(f"[db] host={params['host']} port={params['port']} db={params['dbname']}")
    print(f"[workers] {args.workers}; chunk_size={args.chunk_size}")

    processed = 0
    mapped_pairs = 0
    pending = set()
    max_pending = max(args.workers * 2, 2)

    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["uri", "country"])

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for block in chunks(iter_uris(files), args.chunk_size):
                pending.add(ex.submit(probe_chunk, params, block))
                processed += len(block)

                if len(pending) >= max_pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for fut in done:
                        rows = fut.result()
                        for uri, adm0 in rows:
                            country = country_by_code.get(str(adm0))
                            if country:
                                writer.writerow([str(uri), country])
                                mapped_pairs += 1

                    print(f"\r[progress] candidate URIs submitted={processed:,} mapped pairs={mapped_pairs:,}",
                          end="", flush=True)

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    rows = fut.result()
                    for uri, adm0 in rows:
                        country = country_by_code.get(str(adm0))
                        if country:
                            writer.writerow([str(uri), country])
                            mapped_pairs += 1
                print(f"\r[progress] candidate URIs submitted={processed:,} mapped pairs={mapped_pairs:,}",
                      end="", flush=True)

    print()
    print(f"DONE")
    print(f"candidate URIs processed: {processed:,}")
    print(f"uri-country pairs written: {mapped_pairs:,}")
    print(f"output: {out_path.resolve()}")


if __name__ == "__main__":
    main()
