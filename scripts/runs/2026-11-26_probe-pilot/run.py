import netguard  # noqa: F401  (must stay first: installs the network allow-list before promptchain loads litellm)
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

from promptchain.profiler.governor import GovernorClient, GovernorRefusal, gate_decision, is_resident, snapshot
from promptchain.profiler.latency_probe import RunContext, repo_git_sha, run_latency_probe, summarize
from promptchain.profiler.tool_probe import load_source, run_tool_probe

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
RUN_ID = uuid.uuid4().hex
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
    results = {"run_id": RUN_ID, "before": {"budget": client.budget(), "pressure": client.pressure(), "meters_tail": meters_tail()},
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
            ctx = RunContext(run_id=RUN_ID, git_sha=repo_git_sha(), bfcl_revision=load_source().get("revision"),
                             digest=show.get("digest") or client.digest(tag))
            row.update(status="RAN", capabilities=caps, quant_show=(show.get("details") or {}).get("quantization_level"),
                       digest=ctx.digest, catalog_caps=entry["capabilities"])
            lat = run_latency_probe(client, tag, show=show, capabilities=caps, ctx=ctx, gate=gate)
            row["latency"] = summarize(lat)
            row["latency_calls"] = [{"size": r["input_size"], "repeat": r["repeat"], "warm": r["warm"],
                                     "status": r["status"], "leak": r["thinking_leak"], **r["metrics"]} for r in lat]
            print(f"[{tag}] latency {row['latency']}", flush=True)
            tool = run_tool_probe(client, tag, show=show, capabilities=caps, n_simple=3, n_irrelevance=2,
                                  ctx=ctx, gate=gate)
            row["tool"] = tool
            print(f"[{tag}] tool {tool.get('summary')}", flush=True)
        except GovernorRefusal as e:
            # The governor's own estimate replaces the catalog need; recorded, never retried.
            live = gate_decision(tag, entry.get("mem_gb_at_ctx"), client.budget(), client.pressure(),
                                 governor_est_gb=e.est_gb)
            row["status"] = "DROPPED"
            row["refusal"] = {"code": e.code, "est_gb": e.est_gb, "ceiling_gb": e.ceiling_gb, "msg": str(e),
                              "retry_after_s": e.retry_after_s, "catalog_mem_gb_at_ctx": entry.get("mem_gb_at_ctx"),
                              "live_free_budget_gb": live.free_budget_gb}
            print(f"[{tag}] DROPPED by governor: {e}", flush=True)
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
