import json

with open("dataset/risk-factor/run-15052025/sft/train.v4.sft.context.events.384.candidates.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

for item in data:
    for event in item["answer"]["events"]:
        event["trigger"].pop("left_context", None)
        event["trigger"].pop("right_context", None)
        for argument in event.get("arguments", []):
            argument.pop("left_context", None)
            argument.pop("right_context", None)

with open("dataset/risk-factor/run-15052025/sft/train.v4.sft.events.384.candidates.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")