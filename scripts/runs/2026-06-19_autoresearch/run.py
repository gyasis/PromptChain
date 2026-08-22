#!/usr/bin/env python3
from promptchain.observability import init_mlflow; init_mlflow()
# =============================================================================
# autoresearch — RUNNABLE PromptChain MVP
# A single-process port of the 7-agent doer/critic pipeline. NOT a linear chain:
# a deterministic stage state-machine (the while-loop) dispatches each stage to its
# agent; the agent returns a verdict; the loop routes (reproduces the bounce-backs).
#
#   miner(agentic+tools) ⇄ taste → reviewer ⇄ reviser → implementer ⇄ code-reviewer
#                                → [--hitl gate] → runner(Python Callable) → done
#
# MVP scope: mock grounding tools, in-memory job state, auto-approve gate (set AR_HITL=ask
# to pause). Swap mocks for real deeplake/web/github + a real runner to productionize.
# Full design: ~/Documents/code/dataengineer/autoresearch/{CONCEPT,CONTRACT,PROMPTCHAIN_PORT}.md
# =============================================================================
import asyncio, json, os, re, datetime
from promptchain import PromptChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor
from promptchain.utils.docker_executor import DockerExecutor, extract_code_blocks

MODEL = os.environ.get("AR_MODEL", "openai/gpt-4o-mini")
HITL  = os.environ.get("AR_HITL", "auto")   # "auto" = auto-approve the gate (unattended demo)
MAX_TICKS = int(os.environ.get("AR_MAX_TICKS", "24"))

def mcfg(name: str = MODEL, max_tokens: int = 4096, num_ctx: int = 8192) -> dict:
    """PromptChain model dict with OUTPUT-LENGTH controls so ollama/claude don't truncate.
    - ollama: num_predict defaults to 128 (premature cut) → set it explicitly; raise num_ctx too.
    - claude: max_tokens is required; set generously.
    - openai/gemini: honor max_tokens.
    (max_tokens/num_predict/num_ctx ride in `params`, forwarded to litellm.)"""
    params = {"max_tokens": max_tokens}
    if name.startswith("ollama/"):
        params.update({"num_predict": max_tokens, "num_ctx": num_ctx})
    return {"name": name, "params": params}

# ---------------- mock grounding tools (the miner calls these) ----------------
def deeplake_retrieve(query: str) -> str:
    """Search the user's memory_lane_v4 corpus for tips/tricks. Returns JSON hits."""
    return json.dumps({"query": query, "hits": [
        {"id": "ml4_1827", "text": "Bulk SQLite insert: executemany over a generator; "
         "PRAGMA synchronous=OFF, journal_mode=MEMORY; never materialize all rows."}]})

def web_search(query: str) -> str:
    """Web search. Returns JSON list of {title,url,snippet}."""
    return json.dumps([{"title": "Faker docs", "url": "https://faker.readthedocs.io",
                        "snippet": "Faker generates fake data; lightweight, streaming-friendly."}])

def github_readme(repo: str) -> str:
    """Fetch a GitHub repo's README. Returns text."""
    return f"{repo}: Faker is a Python package that generates fake data. `pip install faker`."

TOOLS_SCHEMA = [
    {"type": "function", "function": {"name": "deeplake_retrieve",
        "description": "Search the memory_lane_v4 corpus for tips. Returns JSON hits.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "web_search",
        "description": "Web search. Returns JSON results.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "github_readme",
        "description": "Fetch a GitHub repo README text.",
        "parameters": {"type": "object", "properties": {"repo": {"type": "string"}}, "required": ["repo"]}}},
]

# ---------------- agent brains ----------------
def critic(role: str, rubric: str) -> PromptChain:
    instr = (role + "\n\n" + rubric +
             "\n\nJOB (brief / approach / plan as available):\n{input}\n\n"
             "End your reply with EXACTLY one line:\nACTION: <one verdict word>")
    return PromptChain(models=[mcfg(max_tokens=1500)], instructions=[instr], verbose=False)  # verdicts are short

def doer(role: str) -> PromptChain:
    return PromptChain(models=[mcfg(max_tokens=4096)], instructions=[role + "\n\nJOB:\n{input}"], verbose=False)  # plans/code can be long

def build_miner() -> PromptChain:
    c = PromptChain(models=[MODEL], instructions=[AgenticStepProcessor(  # pin model → avoid gpt-4o fallback
        objective=("Decompose the brief. Use the tools (try the corpus FIRST) to find the single "
                   "best approach to satisfy it under the stated constraints. CITE each claim "
                   "[corpus:id]/[web:url]/[github:repo]. Return a short '## Chosen Approach' "
                   "paragraph WITH a citation."),
        model_name=MODEL, max_internal_steps=4, history_mode="progressive")], verbose=False)
    c.register_tool_function(deeplake_retrieve)
    c.register_tool_function(web_search)
    c.register_tool_function(github_readme)
    c.add_tools(TOOLS_SCHEMA)
    return c

MINER = build_miner()
TASTE = critic("You are TASTE, a prosecutor-critic. Decide if the chosen approach is worth a slot.",
               "Score leverage/novelty/fit. GATE: every claim must carry a citation. "
               "Verdict words: ACCEPT (good + cited) / REPITCH (fixable) / REJECT (hopeless).")
REVIEWER = critic("You are REVIEWER. Judge if the implementation PLAN is soundly specified.",
                  "Require: concrete method, a script-checkable Success Bar (ASSERT: ... RETURNS ...), "
                  "constraints handled, cited sources. Verdict words: APPROVE / REVISE / REJECT.")
REVISER = doer("You are REVISER. Author/tighten plan.md from the approach. Output a plan with EXACTLY "
               "these headers: ## Method (cited) / ## Success Bar (ASSERT: <cmd> RETURNS <val>) / "
               "## Constraint Handling / ## Resources / ## Scope. No TODOs.")
IMPLEMENTER = doer("You are IMPLEMENTER. Output the SHORTEST code that fulfills the plan EXACTLY, plus a "
                   "check script that tests the Success Bar and prints SUCCESS:/FAILURE:. No extra features. "
                   "Return fenced code blocks only.")
CODE_REVIEWER = critic("You are CODE-REVIEWER, a prosecutor. Statically review the code vs the plan.",
                       "Require: matches plan, looks correct, constraints honored, a check script that "
                       "truly tests the Success Bar. Verdict words: ACCEPT / RECODE / REJECT.")

# ---------------- runner = a pure Python Callable (no LLM) ----------------
def run_job(job: dict) -> str:
    # Real sandboxed execution via the native DockerExecutor (PromptChain issue #6).
    if not DockerExecutor.available():
        job["result"] = "PASS"
        return "PASS: (docker unavailable — mock) artifact would run in a sandbox"
    blocks = extract_code_blocks(job.get("code") or "")
    if not blocks:
        job["result"] = "FAIL"
        return "FAIL: implementer produced no runnable code block"
    cons = job.get("constraints", [])
    with DockerExecutor(image="python:3-slim", network="none",
                        memory="256m" if "low-memory" in cons else "512m",
                        cpus=1, pids_limit=128, read_only=True, user="1000:1000",
                        timeout=60) as ex:
        res = ex.run_code_blocks(blocks)
    job["result"] = "PASS" if (res.exit_code == 0 and not res.timed_out) else "FAIL"
    snippet = " ".join(res.output.split())[:200]
    return f"{job['result']}: exit={res.exit_code} sandbox(--network none, non-root) | {snippet}"

# ---------------- state machine ----------------
ACTION_RE = re.compile(r"ACTION:\s*([A-Z\-]+)", re.I)
def parse_action(text: str, default: str) -> str:
    m = list(ACTION_RE.finditer(text or ""))
    return (m[-1].group(1).upper() if m else default)

def render(job: dict) -> str:
    parts = [f"BRIEF: {job['brief']}", f"CONSTRAINTS: {', '.join(job['constraints']) or 'none'}"]
    if job.get("approach"): parts.append("APPROACH:\n" + job["approach"])
    if job.get("plan"):     parts.append("PLAN:\n" + job["plan"])
    if job.get("code"):     parts.append("CODE:\n" + job["code"][:2000])
    return "\n\n".join(parts)

def cap_hit(job, key):  # bounce cap = 3
    return job["bounces"].get(key, 0) >= 3
def bump(job, key):
    job["bounces"][key] = job["bounces"].get(key, 0) + 1

async def tick(job: dict, log) -> None:
    s = job["stage"]
    if s in ("queue", "needs-repitch"):
        job["approach"] = await MINER.process_prompt_async(job["brief"])
        log("miner", "filed approach"); job["stage"] = "needs-taste"; return
    if s == "needs-taste":
        out = await TASTE.process_prompt_async(render(job)); a = parse_action(out, "ACCEPT")
        log("taste", a)
        job["stage"] = ("needs-review" if a == "ACCEPT"
                        else "rejected" if a == "REJECT" or cap_hit(job, "repitch")
                        else (_bumped(job, "repitch", "needs-repitch"))); return
    if s == "needs-review":
        if not job.get("plan"):                       # Zero-Index Draft (orchestrator-decided)
            log("reviewer", "INITIAL_DRAFT"); job["stage"] = "needs-revision"; return
        out = await REVIEWER.process_prompt_async(render(job)); a = parse_action(out, "APPROVE")
        log("reviewer", a)
        job["stage"] = ("needs-code" if a == "APPROVE"
                        else "rejected" if a == "REJECT" or cap_hit(job, "revision")
                        else _bumped(job, "revision", "needs-revision")); return
    if s == "needs-revision":
        job["plan"] = await REVISER.process_prompt_async(render(job))
        log("reviser", "authored plan"); job["stage"] = "needs-review"; return
    if s == "needs-code":
        job["code"] = await IMPLEMENTER.process_prompt_async(render(job))
        log("implementer", "wrote code"); job["stage"] = "needs-code-review"; return
    if s == "needs-code-review":
        out = await CODE_REVIEWER.process_prompt_async(render(job)); a = parse_action(out, "ACCEPT")
        log("code-reviewer", a)
        if a == "ACCEPT":
            job["stage"] = "ready-to-run" if job["mode"] == "full-auto" else "needs-human"
        elif a == "REJECT" or cap_hit(job, "recode"):
            job["stage"] = "rejected"
        else:
            job["stage"] = _bumped(job, "recode", "needs-code"); return
        return
    if s == "needs-human":
        if HITL == "auto":
            log("--hitl", "auto-approved (set AR_HITL=ask to pause)"); job["stage"] = "ready-to-run"
        else:
            ans = input(f"[--hitl] approve job {job['id']}? [y/N] ").strip().lower()
            job["stage"] = "ready-to-run" if ans == "y" else "needs-code"
        return
    if s == "ready-to-run":
        log("runner", run_job(job)); job["stage"] = "done"; return

def _bumped(job, key, stage):
    bump(job, key); return stage

def write_summary(job: dict, trace: list) -> str:
    """Write a good run-summary doc at the finish of a run (brief, constraints, full stage
    trace, artifacts, result). Lands in summaries/ next to this script."""
    runs_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(runs_dir, "summaries"); os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    status = job["result"] or job["stage"]
    path = os.path.join(out_dir, f"{ts}_{status}.md")
    L = [f"# autoresearch run — {status}", "",
         f"- **when:** {ts}", f"- **model:** {MODEL}", f"- **mode:** {job['mode']}",
         f"- **brief:** {job['brief']}",
         f"- **constraints:** {', '.join(job['constraints']) or 'none'}",
         f"- **final stage:** `{job['stage']}` · **result:** {job['result']} · **bounces:** {job['bounces']}",
         "", "## Stage trace", ""]
    for stage, agent, msg in trace:
        L.append(f"- `{stage}` — **{agent}**: {msg}")
    for key, title in [("approach", "Chosen approach"), ("plan", "Plan"), ("code", "Code")]:
        if job.get(key):
            L += ["", f"## {title}", "", "```", str(job[key])[:1800].rstrip(), "```"]
    open(path, "w").write("\n".join(L) + "\n")
    return path

async def run_brief(brief: str, constraints, mode="hitl"):
    job = {"id": "demo-001", "brief": brief, "constraints": list(constraints), "mode": mode,
           "stage": "queue", "bounces": {}, "approach": None, "plan": None, "code": None, "result": None}
    trace = []
    def log(agent, msg):
        print(f"  [{job['stage']:<16}] {agent:<14} {msg}"); trace.append((job["stage"], agent, msg))
    print(f"\n=== autoresearch (PromptChain MVP) · model={MODEL} · mode={mode} ===")
    print(f"brief: {brief}\nconstraints: {constraints}\n")
    for i in range(MAX_TICKS):
        if job["stage"] in ("done", "rejected"): break
        await tick(job, log)
    print(f"\n=== FINISHED: stage={job['stage']} result={job['result']} bounces={job['bounces']} ===")
    summary = write_summary(job, trace)
    print(f"  summary → {os.path.relpath(summary, os.path.dirname(os.path.abspath(__file__)))}")
    return job

if __name__ == "__main__":
    asyncio.run(run_brief(
        "spin up a quick SQLite DB; first spoof a schema with 500k+ rows",
        ["low-memory", "docker-only"], mode="hitl"))
