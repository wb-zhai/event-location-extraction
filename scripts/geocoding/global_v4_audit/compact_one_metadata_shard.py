#!/usr/bin/env python3

import sys
import time

from compact_event_metadata import process_shard

if len(sys.argv) != 2:
    raise SystemExit(
        "usage: python compact_one_metadata_shard.py SHARD_NUMBER"
    )

i = int(sys.argv[1])

print(f"starting shard-{i:03d}", flush=True)

t0 = time.time()

result = process_shard(i)

dt = time.time() - t0

print(
    f"DONE shard-{i:03d} "
    f"articles={result['articles']:,} "
    f"events={result['events']:,} "
    f"time={dt/60:.2f} min",
    flush=True,
)
