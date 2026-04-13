# /// script
# dependencies = [
#   "fastcoref", "transformers>=4.38,<4.40",
#   "spacy",
#   "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
# ]
# ///

"""Isolated coreference resolution worker.

Reads a JSON list of texts from stdin, writes coref clusters to stdout.
Runs in its own uv environment with pinned transformers to avoid conflicts.
"""

import json
import sys
import warnings

warnings.filterwarnings("ignore")

from fastcoref import FCoref

model = FCoref(device="cpu")

texts = json.loads(sys.stdin.read())
preds = model.predict(texts=texts)

output = []
for pred in preds:
    clusters = []
    for strings, offsets in zip(
        pred.get_clusters(as_strings=True),
        pred.get_clusters(as_strings=False),
    ):
        clusters.append({"strings": strings, "offsets": [[s, e] for s, e in offsets]})
    output.append(clusters)

sys.stdout.write(json.dumps(output))
