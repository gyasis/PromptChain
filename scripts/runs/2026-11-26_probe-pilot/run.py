import json
import os
import sys
import time
from pathlib import Path

import promptchain.profiler.netguard as netguard
from promptchain.profiler.governor import GovernorClient, gate_decision, is_resident, snapshot
from promptchain.profiler.latency_probe import summarize, run_latency_probe
from promptchain.profiler.tool_probe import run_tool_probe

from promptchain.observability import init_mlflow; init_mlflow()

COHORT = [
    "ornith-1.5-9b:latest",
    "gemma4:12b",
    "gpt-oss:20b",
    "ornith-1.5-35b:latest",
    "qwen3.8:27b-mtp-q4_K_M",
    "hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL",
]
BUDGET_S = 45 * 60
OUT = Path(os.environ.get("PROBE_OUT", "/tmp/probe-pilot-results.json"))
CATALOG = json.load(open(os.path.expanduser("~/.config/ollama-cloud/local-models.json")))
CATALOG_MD = open(os.path.expanduser("~/.config/ollama-cloud/local-models.md")).read()
METERS = Path(os.path.expanduser("~/.local/share/llm-snitch/meters.jsonl"))


def meters_tail(n=6):
    return [json.loads(x) for x in METERS.read_text().splitlines()[-n:]] if METERS.exists() else []


def wait_until_gone(client, tag, limit_s=200):
    """Wait for our own previous model to fall out of residency (governor keep-alive); never evict it."""
    t0 = time.time()
    while time.time() - t0 < limit_s and is_resident(tag, client.pressure()):
        time.sleep(10)


def main():
    t_start = time.time()
    client = GovernorClient()
    results = {"before": {"budget": client.budget(), "pressure": client.pressure(), "meters_tail": meters_tail()},
               "models": {}}
    by_name = {m["name"]: m for m in CATALOG["models"]}
    prev = None
    for tag in COHORT:
        if time.time() - t_start > BUDGET_S:
            results["models"][tag] = {"status": "NOT RUN: wall-clock budget reached"}
            continue
        if f"`{tag}`" not in CATALOG_MD or tag not in by_name:
            results["models"][tag] = {"status": "SKIPPED: tag not in local-models.md"}
            continue
        if prev:
            wait_until_gone(client, prev)
        entry = by_name[tag]
        budget, pressure = client.budget(), client.pressure()
        gate = gate_decision(tag, entry.get("mem_gb_at_ctx"), budget, pressure)
        row = {"gate": gate.to_dict(), "governor": snapshot(budget, pressure)}
        results["models"][tag] = row
        print(f"[{tag}] gate: {gate.admitted} {gate.reason}", flush=True)
        if not gate.admitted:
            row["status"] = "DROPPED"
            continue
        t0 = time.time()
        try:
            show = client.show(tag)
            caps = show.get("capabilities") or []
            quant = (show.get("details") or {}).get("quantization_level")
            row.update(status="RAN", capabilities=caps, quant=quant, catalog_caps=entry["capabilities"])
            lat = run_latency_probe(client, tag, quant=quant or tag.rsplit(":", 1)[-1], capabilities=caps, gate=gate)
            row["latency"] = summarize(lat)
            row["latency_calls"] = [{"size": r["input_size"], "warm": r["warm"], **r["metrics"]} for r in lat]
            print(f"[{tag}] latency {row['latency']}", flush=True)
            tool = run_tool_probe(client, tag, quant=quant or tag.rsplit(":", 1)[-1], capabilities=caps, gate=gate)
            row["tool"] = tool
            print(f"[{tag}] tool {tool.get('summary')}", flush=True)
        except Exception as e:  # recorded, never retried
            row["status"] = "FAILED"
            row["error"] = repr(e)
            print(f"[{tag}] FAILED {e!r}", flush=True)
        row["wall_s"] = time.time() - t0
        prev = tag
        OUT.write_text(json.dumps(results, indent=1, default=str))
    results["after"] = {"budget": client.budget(), "pressure": client.pressure(), "meters_tail": meters_tail()}
    results["total_wall_s"] = time.time() - t_start
    results["blocked_connections"] = netguard.blocked
    OUT.write_text(json.dumps(results, indent=1, default=str))
    print("done", results["total_wall_s"], "blocked:", netguard.blocked)


if __name__ == "__main__":
    sys.exit(main())
