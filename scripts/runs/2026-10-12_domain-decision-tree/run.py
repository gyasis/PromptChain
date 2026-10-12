"""domain-decision-tree — 2-level Jev-style decision tree in PromptChain, real engines.

Level 1 (new, unmeasured): which domain is this text, clinical or legal?
Level 2 (measured in the harness-lab bake-off): clinical specialty / legal clause type,
using the exact questions + labelled items from harness-lab/bench/decisions/.

    python run.py --backend tev1          # systemone tev1:4b (default backend)
    python run.py --backend hunch         # hunch, mistral-small:22b (the bake-off hunch model)
"""
from promptchain.observability import init_mlflow
init_mlflow()

import argparse
import asyncio
import json
import time
from pathlib import Path

from promptchain import PromptChain
from promptchain.utils.decision_step import SystemOneBackend, decision, hunch_backend, route_on

BENCH = Path.home() / "Documents/code/harness-lab/bench/decisions"
ACCEPT, REVIEW = 0.9, 0.6  # PLACEHOLDER thresholds, not measured for this task


class Recording:
    """Wraps a backend and keeps every answer, so the run can show the confidence per level."""

    def __init__(self, inner):
        self.inner, self.name, self.model, self.log = inner, inner.name, inner.model, []

    async def decide(self, state, questions):
        reply = await self.inner.decide(state, questions)
        for q, a in reply["answers"].items():
            self.log.append((q, a.get("choice"), a.get("confidence")))
        return reply


def load(name, per_label=2):
    rows = [json.loads(line) for line in open(BENCH / f"{name}.jsonl")]
    picked, seen = [], {}
    for r in rows:
        gold = next(iter(r["gold"].values()))
        if seen.get(gold, 0) < per_label:
            seen[gold] = seen.get(gold, 0) + 1
            picked.append(r)
    return picked


def build(backend, clinical_q, legal_q):
    leaf = lambda dom: {lab: (lambda s, d=dom, l=lab: f"{d} > {l}") for lab in
                        next(iter((clinical_q if dom == "clinical" else legal_q).values()))["criteria"]}

    def subtree(dom, q):
        qname = next(iter(q))
        return PromptChain(models=[], instructions=[
            decision(q, backend, name=f"{dom}_detail"),
            route_on(qname, accept=ACCEPT, review=REVIEW, on_accept=leaf(dom),
                     on_review=lambda s, d=dom: f"{d} > REVIEW",
                     on_escalate=lambda s, d=dom: f"{d} > ESCALATE"),
        ])

    domain_q = {"domain": {"type": "choice",
                           "instructions": "Which professional domain does this text belong to?",
                           "criteria": {"clinical": "medical dictation: patients, procedures, diagnoses, imaging",
                                        "legal": "contract or agreement language: clauses, parties, obligations"}}}
    return PromptChain(models=[], instructions=[
        decision(domain_q, backend, name="domain"),
        route_on("domain", accept=ACCEPT, review=REVIEW,
                 on_accept={"clinical": subtree("clinical", clinical_q),
                            "legal": subtree("legal", legal_q)},
                 on_review=lambda s: "REVIEW (domain unclear)",
                 on_escalate=lambda s: "ESCALATE (domain unclear)"),
    ])


CROSSOVER = [
    "The Physician shall maintain professional liability insurance of not less than $1,000,000 "
    "per occurrence and shall notify the Hospital within ten days of any malpractice claim.",
    "Patient signed informed consent for laparoscopic cholecystectomy after risks, benefits and "
    "alternatives were explained; consent form witnessed and filed.",
    "Upon any Event of Default, all amounts owed for the patient's treatment shall become "
    "immediately due and payable.",
]


async def main(which):
    inner = (SystemOneBackend(model="tev1:4b") if which == "tev1"
             else hunch_backend(model="mistral-small:22b"))
    backend = Recording(inner)
    clin, leg = load("clinical"), load("legal")
    chain = build(backend, clin[0]["questions"], leg[0]["questions"])
    items = ([("clinical", r) for r in clin] + [("legal", r) for r in leg]
             + [("crossover", {"state": t, "gold": {}}) for t in CROSSOVER])
    print(f"backend={backend.name} model={backend.model}  thresholds accept={ACCEPT} review={REVIEW}\n")
    right = total = 0
    t0 = time.time()
    for dom, r in items:
        backend.log.clear()
        out = await chain.process_prompt_async(r["state"])
        gold = f"{dom} > {next(iter(r['gold'].values()))}" if r["gold"] else "(no gold)"
        ok = "" if not r["gold"] else ("OK " if out == gold else "XX ")
        if r["gold"]:
            total += 1
            right += out == gold
        trace = "  ".join(f"{q}={c}@{conf:.3f}" for q, c, conf in backend.log)
        print(f"{ok:3}{out:42} gold: {gold:38} {trace}\n     text: {r['state'][:90]}")
    print(f"\nlabelled items routed to the exact gold leaf: {right}/{total}   wall {time.time()-t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["tev1", "hunch"], required=True)
    asyncio.run(main(ap.parse_args().backend))
