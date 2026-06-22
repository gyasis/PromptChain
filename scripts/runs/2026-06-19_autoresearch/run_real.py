#!/usr/bin/env python3
from promptchain.observability import init_mlflow; init_mlflow()
# =============================================================================
# autoresearch — REAL run (ZERO mocks): ollama qwen + real grounding tools + real docker.
#   model:   ollama/qwen3:32b (reasoning model; <think> stripped; generous num_predict)
#   deeplake: REAL — memory_lane_v4 via the deeplakesearch .venv (ada-002 + S3)
#   web:      REAL — DuckDuckGo via ddgs
#   github:   REAL — GitHub API README fetch (public, no key)
#   runner:   REAL — DockerExecutor (sandboxed)
# Grounding tool calls are ORCHESTRATED in Python (deterministic + reliable on a local
# model) — qwen does the reasoning/synthesis/verdicts. Every call is real.
# =============================================================================
import asyncio, json, os, re, datetime, subprocess, urllib.request, base64
from promptchain import PromptChain
from promptchain.utils.docker_executor import DockerExecutor, extract_code_blocks

# Ollama CLOUD qwen via the OpenAI-compatible endpoint (more powerful + faster than local Mac models).
MODEL = os.environ.get("AR_MODEL", "openai/gpt-oss:120b")  # clean default for research — qwen emits heavy <think> (use AR_MODEL to override)
OLLAMA_CLOUD = {"api_base": "https://ollama.com/v1", "api_key": os.environ.get("OLLAMA_API_KEY", "")}
HITL = os.environ.get("AR_HITL", "auto")
SHADOW = os.environ.get("AR_SHADOW") == "1"   # advisory: critics log their REAL verdict but force-advance (collect verdict-vs-runner data)
MAX_TICKS = int(os.environ.get("AR_MAX_TICKS", "24"))
DEEPLAKE_DIR = os.path.expanduser("~/Documents/code/deeplakesearch")
DEEPLAKE_PY = os.path.join(DEEPLAKE_DIR, ".venv/bin/python")

def mcfg(max_tokens: int = 3000, name: str = MODEL) -> dict:
    # qwen3.5 is a REASONING model — big max_tokens so <think> doesn't eat the answer.
    params = {"max_tokens": max_tokens}
    if name.startswith("openai/"):
        params.update(OLLAMA_CLOUD)                                       # → Ollama Cloud (qwen/gpt-oss/glm/kimi/... via ollama.com OpenAI-compat)
    elif name.startswith("ollama/"):
        params.update({"num_predict": max_tokens, "num_ctx": 8192})       # local ollama
    return {"name": name, "params": params}

_THINK = re.compile(r"<think>.*?</think>", re.S)
def strip_think(t: str) -> str:
    return _THINK.sub("", t or "").strip()

# ---------------- REAL grounding tools (orchestrated; zero mocks) ----------------
def deeplake_retrieve(query: str, n: int = 5) -> str:
    """REAL: query memory_lane_v4 via the deeplakesearch .venv (ada-002 embed + S3)."""
    code = "import json, main; print(json.dumps(main.retrieve_context(%r, n_results=%d))[:3500])" % (query, n)
    try:
        p = subprocess.run([DEEPLAKE_PY, "-c", code], cwd=DEEPLAKE_DIR,
                           capture_output=True, text=True, timeout=120)
        return p.stdout.strip() or ("deeplake-error: " + (p.stderr.strip()[-300:] or "empty"))
    except Exception as e:
        return f"deeplake-error: {e}"

def web_search(query: str, n: int = 5) -> str:
    """REAL: DuckDuckGo via ddgs (keyless)."""
    try:
        from ddgs import DDGS
        res = list(DDGS().text(query, max_results=n))
        return json.dumps([{"title": r.get("title"), "url": r.get("href"),
                            "snippet": (r.get("body") or "")[:200]} for r in res])
    except Exception as e:
        return f"web-error: {e}"

def github_readme(repo: str) -> str:
    """REAL: fetch a repo's README via the GitHub API (public, no key)."""
    try:
        req = urllib.request.Request(
            f"https://api.github.com/repos/{repo}/readme",
            headers={"Accept": "application/vnd.github+json", "User-Agent": "autoresearch"})
        with urllib.request.urlopen(req, timeout=25) as r:
            data = json.load(r)
        return base64.b64decode(data.get("content", "")).decode("utf-8", "ignore")[:1800]
    except Exception as e:
        return f"github-error: {e}"

_GH = re.compile(r"github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)")
def miner_gather(brief: str):
    """Run all three REAL tools, assemble cited evidence. Returns (evidence, grounded).
    F2: grounded=False if BOTH corpus AND web errored — so the pipeline REJECTS instead of
    synthesizing an approach from error-strings dressed up as citations."""
    corpus = deeplake_retrieve(brief)
    web = web_search(brief)
    grounded = not (corpus.startswith("deeplake-error") and web.startswith("web-error"))
    ev = [f"[corpus] memory_lane_v4:\n{corpus[:1400]}", f"[web] DuckDuckGo:\n{web[:1200]}"]
    m = _GH.search(web)
    if m:
        repo = m.group(1).rstrip(".git")
        ev.append(f"[github:{repo}] README:\n{github_readme(repo)[:1200]}")
    return "\n\n".join(ev), grounded

# ---- atelier (Mac Studio) — the GPU "house": REUSE the existing io.macstudio.hub.mlxlm sidecar ----
ATELIER_URL = os.environ.get("AR_ATELIER_URL", "http://192.168.0.159:8773/v1")   # OpenAI-compatible MLX-LM sidecar (Apple GPU / Metal)
ATELIER_MODEL = os.environ.get("AR_ATELIER_MODEL", "/Users/gyasisutton/models/mlx-llm/Qwen2.5-0.5B-Instruct-4bit")  # full id (alias broken offline)

def atelier_status(timeout=3):
    """Probe the Mac Studio MLX GPU sidecar. Returns (available: bool, detail: str)."""
    try:
        with urllib.request.urlopen(ATELIER_URL.rstrip("/") + "/models", timeout=timeout) as r:
            ids = [m.get("id", "") for m in json.load(r).get("data", [])]
        return True, (", ".join(i.split("/")[-1] for i in ids)[:80] or "no models")
    except Exception as e:
        return False, f"offline ({str(e)[:40]})"

def atelier_model_call(prompt, max_tokens=512, model=None):
    """REAL Apple-GPU inference via the atelier MLX sidecar (OpenAI-compatible)."""
    body = json.dumps({"model": model or ATELIER_MODEL,
                       "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(ATELIER_URL.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["choices"][0]["message"]["content"]

# ---- atelier MANAGER (governor) — lean on it to know what FITS / the house's limits ----
GOVERNOR_URL = os.environ.get("AR_GOVERNOR_URL", "http://192.168.0.159:8799")   # io.macstudio.hub.governor (memory/admission API)
def governor_budget(timeout=3):
    """Ask the atelier manager what fits. Returns its /budget dict (free_budget_gb, budget_gb, num_parallel) or None."""
    try:
        with urllib.request.urlopen(GOVERNOR_URL.rstrip("/") + "/budget", timeout=timeout) as r:
            return json.load(r)
    except Exception:
        return None
def governor_fit(gb, timeout=3):
    """True if the manager has >= gb free; False if not; None if the manager is unreachable."""
    b = governor_budget(timeout)
    return None if b is None else (float(b.get("free_budget_gb", 0)) >= float(gb))

# ---- Execution-environment CONTRACT (single source of truth) ----
# Both the IMPLEMENTER (codes TO it) and CODE-REVIEWER (checks AGAINST it) reference this.
# Keep in sync with run_job()'s DockerExecutor config below: change the box → change ONE string,
# and both prompts (and any new domain) stay correct. This is what makes the prompts environment-
# accurate rather than hardcoded "stdlib-only as law".
SANDBOX_CONTRACT = (
    "EXECUTION ENVIRONMENT (your code runs here — code TO this contract):\n"
    "- IMAGE: python:3-slim. AVAILABLE: the Python STANDARD LIBRARY ONLY (csv, json, sqlite3, hashlib, "
    "statistics, collections, itertools, re, math, random, datetime, ...). NO third-party packages and NO "
    "pip (no network EXCEPT the atelier GPU endpoint in the GPU line below) — importing any non-stdlib module "
    "(pandas/numpy/scipy/sklearn/requests/datasketch/...) raises ImportError and FAILS the run.\n"
    "- EXECUTION: each ```python block runs in ISOLATION as its own file; a cross-block `from mymod import X` "
    "raises ModuleNotFoundError → put EVERYTHING in ONE block.\n"
    "- I/O: no CLI args, no stdin, read-only filesystem (no files). Do NOT read sys.argv/stdin/files — hardcode "
    "a tiny INLINE sample.\n"
    "- MODEL INFERENCE — two CALL lanes (stdlib urllib, POST /chat/completions; network to the host you call IS "
    "enabled; NEVER load/import models locally — no torch/GPU/pip; NEVER use 'gpt-4o'/'gpt-4o-mini'):\n"
    "    (A) BEST for LLM tasks (reasoning/generation/classification) — OLLAMA CLOUD https://ollama.com/v1, "
    "model 'gpt-oss:120b' or 'glm-5' (CLEAN; do NOT use qwen — it emits heavy <think>). Auth header "
    "'Authorization: Bearer ' + os.environ['OLLAMA_API_KEY'] (the key is already in your container env).\n"
    "    (B) Local/offline small model — atelier MLX http://192.168.0.159:8773/v1, model EXACTLY "
    "'/Users/gyasisutton/models/mlx-llm/Qwen2.5-0.5B-Instruct-4bit' (keyless LAN).\n"
    "- DEFINITION OF DONE: one self-contained ```python block, stdlib-only, with an inline check that prints "
    "SUCCESS: or FAILURE: and exits non-zero on failure."
)

# ---------------- agents (qwen sequential; thinking stripped) ----------------
async def ask(chain: PromptChain, prompt: str) -> str:
    return strip_think(await chain.process_prompt_async(prompt))

def critic(role: str, rubric: str) -> PromptChain:
    instr = (role + "\n\n" + rubric + "\n\nJOB:\n{input}\n\n"
             "End your reply with EXACTLY one line:\nACTION: <one verdict word>")
    return PromptChain(models=[mcfg(max_tokens=4000)], instructions=[instr], verbose=False)   # room for <think> + a final ACTION line

def doer(role: str) -> PromptChain:
    return PromptChain(models=[mcfg(max_tokens=8000)], instructions=[role + "\n\nJOB:\n{input}"], verbose=False)  # think + long output

MINER = doer("You are MINER. Given the brief and the REAL research evidence below (corpus + web + "
             "github, each cited), choose the single best approach. Output '## Chosen Approach' "
             "(2-4 sentences) citing evidence tags [corpus]/[web:url]/[github:repo]. Invent no sources.")
TASTE = critic("You are TASTE — a DEFENSE ATTORNEY for ideas, not a prosecutor.",
               "Default = ACCEPT. Only bounce if you can NAME a concrete disqualifying flaw: zero evidence/citations = REPITCH; "
               "fundamentally off-brief or no leverage = REJECT. A merely-imperfect-but-plausible, cited idea ACCEPTS. "
               "Do NOT invent reasons to bounce. ACCEPT/REPITCH/REJECT.")
REVIEWER = critic("You are REVIEWER — a DEFENSE ATTORNEY for the plan. Review via 3 lenses: "
                  "(1) Skeptic: will it fail? (2) Rule-follower: constraints honored? (3) Mentor: does it meet the bar?",
                  "Default = APPROVE. REVISE only if you QUOTE a concrete BLOCKING gap (NO script-checkable Success Bar, "
                  "or a Method step that genuinely cannot be implemented). Nits/improvements go in a one-line 'suggestions:' "
                  "note and still APPROVE. If you cannot cite a blocking gap, you MUST APPROVE. Never invent a reason to REVISE. APPROVE/REVISE/REJECT.")
REVISER = doer("You are REVISER. Author/upgrade plan.md with EXACTLY these headers: ## Method (cited) / "
               "## Success Bar / ## Constraint Handling / ## Resources / ## Scope. If REVIEWER FEEDBACK is present, "
               "ADDRESS EVERY point it raises. The Success Bar MUST be a REAL runnable check — a concrete shell/python "
               "command that runs the artifact on a tiny inline sample and asserts on its ACTUAL output — NOT a "
               "pseudo-function signature. No TODOs. "
               "HOUSE GPU: a model is available as a CALL — prefer OLLAMA CLOUD (https://ollama.com/v1, model gpt-oss:120b or "
               "glm-5 — CLEAN) for LLM-quality tasks, or the local atelier endpoint (http://192.168.0.159:8773/v1) for small/offline. "
               "For ANY 'use/run/call a model' need, PLAN A CALL to it "
               "(list it under ## Resources; the Success Bar runs the artifact which calls it + asserts on the response). "
               "Do NOT plan to train / fine-tune / distill / pip-install a model — the sandbox cannot, and a call suffices. "
               "Only plan training if the brief is EXPLICITLY about training a model from scratch.")
IMPLEMENTER = doer("You are IMPLEMENTER. Output the SHORTEST python that fulfills the plan, with an inline "
                   "check that prints SUCCESS:/FAILURE:. Your code MUST satisfy this contract:\n\n" + SANDBOX_CONTRACT +
                   "\n\nIf the plan suggests a third-party library, re-implement that approach in pure stdlib — do NOT import it.")
CODE_REVIEWER = critic("You are CODE-REVIEWER — a DEFENSE ATTORNEY doing static review via 3 lenses "
                       "(Skeptic: will it crash? Rule-follower: constraints? Mentor: does the check test the Success Bar?). "
                       "The sandboxed runner is the REAL gate, so favor ACCEPT.",
                       "Default = ACCEPT. RECODE only if you QUOTE a concrete BLOCKING defect: a syntax error, a MISSING "
                       "check script, a hard CONSTRAINT violation, or ANY violation of the EXECUTION ENVIRONMENT contract below "
                       "(non-stdlib import, multi-block cross-import, or reading sys.argv/stdin/files) — those WILL crash. "
                       "GATE_INTEGRITY (BLOCKING): the check MUST honestly exercise the plan's ## Success Bar — RECODE if it is "
                       "HOLLOW/hardcoded (compares a literal to itself, prints SUCCESS without computing the real result) or WEAKER "
                       "than the Success Bar (a 'success-bar delta'). A trivial always-pass is a RECODE, NEVER an ACCEPT. "
                       "ALLOWED, do NOT recode: a stdlib urllib call to the atelier GPU endpoint (192.168.0.159:8773) for "
                       "model inference is EXPECTED and correct — never flag it as a network/isolation violation.\n\n"
                       + SANDBOX_CONTRACT +
                       "\n\nStylistic nits → 'suggestions:' note + ACCEPT. If you cannot cite a crash-level/blocking defect, you MUST ACCEPT. ACCEPT/RECODE/REJECT.")

def run_job(job: dict) -> str:
    if not DockerExecutor.available():
        job["result"] = "ERROR"; job["run_output"] = "docker unavailable — not executed"
        return "ERROR: docker unavailable — code NOT executed (NOT a pass)"  # F1: fail-closed, never fake PASS
    blocks = extract_code_blocks(job.get("code") or "")
    if not blocks:
        job["result"] = "FAIL"; job["run_output"] = "no runnable code block extracted"
        return "FAIL: no runnable code block"
    code_txt = job.get("code") or ""
    atelier_call = ("8773" in code_txt) or ("192.168.0.159" in code_txt) or ("ATELIER" in code_txt.upper())
    cloud_call = ("ollama.com" in code_txt.lower()) or ("OLLAMA_API_KEY" in code_txt)
    needs_net = atelier_call or cloud_call
    net = "bridge" if needs_net else "none"          # model-call briefs reach atelier (LAN) or ollama.com (cloud); else fully isolated
    extra = ["-e", f"OLLAMA_API_KEY={os.environ.get('OLLAMA_API_KEY', '')}"] if cloud_call else []   # key via ENV, never in the saved code
    with DockerExecutor(image="python:3-slim", network=net, memory="512m",
                        cpus=1, pids_limit=128, read_only=True, user="1000:1000",
                        timeout=120 if needs_net else 60, extra_run_args=extra) as ex:
        res = ex.run_code_blocks(blocks)
    job["result"] = "PASS" if (res.exit_code == 0 and not res.timed_out) else "FAIL"
    job["run_exit"] = res.exit_code; job["run_output"] = res.output; job["run_timed_out"] = res.timed_out  # full docker stdout/stderr (+ errors)
    lane = " →atelier-GPU" if atelier_call else (" →ollama-cloud" if cloud_call else "")
    return f"{job['result']}: exit={res.exit_code} sandbox(--network {net}{lane}) | {' '.join(res.output.split())[:200]}"

ACTION_RE = re.compile(r"ACTION:\s*([A-Z\-]+)", re.I)
def parse_action(t, default):
    m = list(ACTION_RE.finditer(t or "")); return (m[-1].group(1).upper() if m else default)
def render(job):
    p = [f"BRIEF: {job['brief']}", f"CONSTRAINTS: {', '.join(job['constraints']) or 'none'}"]
    if job.get("evidence"): p.append("EVIDENCE:\n" + job["evidence"][:2500])
    if job.get("approach"): p.append("APPROACH:\n" + job["approach"])
    if job.get("plan"): p.append("PLAN:\n" + job["plan"])
    if job.get("code"): p.append("CODE:\n" + job["code"][:1800])
    if job.get("last_review"): p.append("REVIEWER FEEDBACK — you MUST address every point:\n" + job["last_review"][:1500])
    if job.get("last_coderev"): p.append("CODE-REVIEWER FEEDBACK — fix exactly these:\n" + job["last_coderev"][:1500])
    return "\n\n".join(p)
def cap(job, k): return job["bounces"].get(k, 0) >= 3
def bumped(job, k, st): job["bounces"][k] = job["bounces"].get(k, 0) + 1; return st

# ---- integrity gates (deterministic, ported from the file-bus GATE_INTEGRITY) ----
# Sandbox blockers: things the runner (python:3-slim, --network none, stdlib, 60s, no GPU) CANNOT do.
# TRUE hardware/training blockers only — these can't be a model-CALL (inference IS doable in-sandbox via the
# atelier endpoint over the network). These remain NEEDS_GPU = build-a-Mac-container last resort / research.
_UNVERIFIABLE_RE = re.compile(
    r"\b(docker\s+(?:pull|run|build)|pip\s+install|apt(?:-get)?\s+install|conda\s+install|"
    r"nvidia|cuda|import\s+torch|import\s+tensorflow|\.fit\(|training\s+loop|fine[- ]?tun|distill)", re.I)
# Beyond-the-house signals → INFEASIBLE (the Mac is 64 GB unified, Apple-GPU only; no NVIDIA/A100/cluster).
_INFEASIBLE_RE = re.compile(
    r"\b(a100|h100|multi[- ]?gpu|gpu\s+cluster|\btpu\b|distributed\s+train|pre[- ]?train|"
    r"(?:70|72|110|120|175|405)\s?b\b|petaflop|hpc\s+cluster)", re.I)

def _success_bar(plan):
    m = re.search(r"##\s*Success Bar\s*(.+?)(?:\n##|\Z)", plan or "", re.S | re.I)
    return (m.group(1).strip() if m else "")

def _sandbox_unverifiable(plan):
    """Return the matched blocker token if the plan's Success Bar can't honestly run in the sandbox, else ''."""
    m = _UNVERIFIABLE_RE.search(_success_bar(plan) or (plan or ""))
    return m.group(0) if m else ""

def _hollow_check(code, plan):
    """True if the check is a trivial always-pass that doesn't exercise a non-trivial Success Bar."""
    blocks = extract_code_blocks(code) or []
    body = "\n".join((b[-1] if isinstance(b, (tuple, list)) else str(b)) for b in blocks) or (code or "")
    lines = [l for l in body.splitlines() if l.strip() and not l.strip().startswith("#")]
    sb = _success_bar(plan)
    nontrivial_bar = len(sb) > 40 or bool(re.search(r"assert|==|grep|\.py\b|rows|count|ratio|\bexit\b", sb, re.I))
    return len(lines) <= 5 and bool(re.search(r"SUCCESS", body, re.I)) and nontrivial_bar

async def tick(job, log):
    s = job["stage"]
    if s in ("queue", "needs-repitch"):
        log("miner", "gathering REAL evidence (deeplake -> web -> github)...")
        job["evidence"], grounded = miner_gather(job["brief"])
        if not grounded:                                                         # F2: no real evidence -> reject, don't fake citations
            log("miner", "REJECT — grounding failed (corpus AND web both errored)")
            job["result"] = "ERROR"; job["stage"] = "rejected"; return
        job["approach"] = await ask(MINER, render(job)); log("miner", "synthesized approach")
        job["stage"] = "needs-taste"; return
    if s == "needs-taste":
        a = parse_action(await TASTE.process_prompt_async(render(job)), "REPITCH")  # parse RAW; F4 default = bounce
        log("taste", a + (" [shadow->accept]" if SHADOW and a != "ACCEPT" else ""))
        if SHADOW: a = "ACCEPT"
        job["stage"] = ("needs-review" if a == "ACCEPT" else "rejected" if a == "REJECT" or cap(job, "repitch") else bumped(job, "repitch", "needs-repitch")); return
    if s == "needs-review":
        if not job.get("plan"):
            log("reviewer", "INITIAL_DRAFT"); job["stage"] = "needs-revision"; return
        bigask = _INFEASIBLE_RE.search(_success_bar(job["plan"]) or job["plan"] or "")
        if bigask:                                        # beyond the house — the researcher KNOWS its limit
            _gb = governor_budget()
            cap = f"{_gb.get('budget_gb',64):.0f}GB unified" if _gb else "64GB unified"
            log("reviewer", f"INFEASIBLE — needs '{bigask.group(0)}', beyond the house (Mac = {cap}, Apple-GPU only; no NVIDIA/A100/cluster). Delivering the researched plan + the hardware it requires; NOT attempted, NOT faked.")
            job["result"] = "INFEASIBLE"; job["stage"] = "done"; return
        blocker = _sandbox_unverifiable(job["plan"])      # gate: bar needs TRAINING/install/hardware (NOT a model-call)
        if blocker:
            _gb = governor_budget()                       # lean on the atelier MANAGER for what fits
            budget = f"manager: {_gb.get('free_budget_gb',0):.0f}GB free of {_gb.get('budget_gb',0):.0f}" if _gb else "manager offline"
            log("reviewer", f"NEEDS-GPU — Success Bar needs '{blocker}' (real GPU training/install; inference is a sandbox→atelier call). atelier [{budget}] — load a FITTING model (ask governor /estimate+/admit); if it won't fit → INFEASIBLE. Plan delivered; NOT a fake pass.")
            job["result"] = "NEEDS_GPU"; job["stage"] = "done"; return
        rv = await REVIEWER.process_prompt_async(render(job)); job["last_review"] = strip_think(rv)  # parse RAW; clean copy for feedback
        a = parse_action(rv, "REVISE")
        log("reviewer", a + (" [shadow->approve]" if SHADOW and a != "APPROVE" else ""))
        if SHADOW: a = "APPROVE"                              # F4: parse-fail -> bounce, never fake-APPROVE
        job["stage"] = ("needs-code" if a == "APPROVE" else "rejected" if a == "REJECT" or cap(job, "revision") else bumped(job, "revision", "needs-revision")); return
    if s == "needs-revision":
        job["plan"] = await ask(REVISER, render(job)); log("reviser", "authored plan"); job["stage"] = "needs-review"; return
    if s == "needs-code":
        raw = await IMPLEMENTER.process_prompt_async(render(job)); job["code"] = raw   # keep RAW: qwen may emit code inside <think>
        log("implementer", f"wrote code ({len(extract_code_blocks(raw))} block(s))"); job["stage"] = "needs-code-review"; return
    if s == "needs-code-review":
        if not extract_code_blocks(job.get("code") or ""):                      # guard: no code -> recode, don't rubber-stamp
            job["last_coderev"] = "No runnable code block found. Emit the FULL code in a fenced python block."
            log("code-reviewer", "RECODE (no code block)")
            job["stage"] = "rejected" if cap(job, "recode") else bumped(job, "recode", "needs-code"); return
        if _hollow_check(job.get("code") or "", job.get("plan") or ""):          # GATE_INTEGRITY: deterministic, catches trivial always-pass stubs
            job["last_coderev"] = "GATE_INTEGRITY: hollow check — it prints SUCCESS without exercising the plan's Success Bar. Implement the REAL check that computes and asserts the actual result."
            log("code-reviewer", "RECODE (GATE_INTEGRITY: hollow check)")
            job["stage"] = "rejected" if cap(job, "recode") else bumped(job, "recode", "needs-code"); return
        cr = await CODE_REVIEWER.process_prompt_async(render(job)); job["last_coderev"] = strip_think(cr)  # parse RAW; clean copy for feedback
        a = parse_action(cr, "RECODE")
        log("code-reviewer", a + (" [shadow->accept]" if SHADOW and a != "ACCEPT" else ""))
        if SHADOW: a = "ACCEPT"                         # F4: parse-fail -> bounce, never fake-ACCEPT
        if a == "ACCEPT": job["stage"] = "ready-to-run" if job["mode"] == "full-auto" else "needs-human"
        elif a == "REJECT" or cap(job, "recode"): job["stage"] = "rejected"
        else: job["stage"] = bumped(job, "recode", "needs-code")
        return
    if s == "needs-human":
        if HITL == "auto":
            log("--hitl", "auto-approved"); job["stage"] = "ready-to-run"
        else:                                                                    # F3: ask mode no longer silently auto-approves
            log("--hitl", "PAUSED — AR_HITL=ask: manual approval required, NOT auto-advancing"); job["_paused"] = True
        return
    if s == "ready-to-run":
        log("runner", run_job(job)); job["stage"] = "done"; return

def write_summary(job, trace):
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "summaries"); os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(out_dir, f"{ts}_REAL_{job['result'] or job['stage']}.md")
    L = [f"# autoresearch REAL run — {job['result'] or job['stage']}", "",
         f"- when: {ts}", f"- model: {MODEL}", f"- mode: {job['mode']}", f"- brief: {job['brief']}",
         f"- final: stage=`{job['stage']}` result={job['result']} bounces={job['bounces']}",
         "", "## Stage trace", ""] + [f"- `{st}` — **{ag}**: {m}" for st, ag, m in trace]
    for k, t in [("evidence", "Evidence (real tools)"), ("approach", "Chosen approach"), ("plan", "Plan"), ("code", "Code")]:
        if job.get(k): L += ["", f"## {t}", "", "```", str(job[k])[:2200].rstrip(), "```"]
    open(path, "w").write("\n".join(L) + "\n"); return path

_RAIL = ["miner", "taste", "reviewer", "reviser", "implementer", "code-reviewer", "runner"]
def _rich_ok():
    if os.environ.get("AR_RICH") == "0":
        return False
    try:
        import rich  # noqa: F401
        return sys.stdout.isatty() or os.environ.get("AR_RICH") == "1"
    except Exception:
        return False
def _rich_view(job, trace):
    from rich.panel import Panel
    from rich.console import Group
    from rich.text import Text
    acted = {a for (_s, a, _m) in trace}
    last = trace[-1][1] if trace else ""
    seg = []
    for a in _RAIL:
        if a == last and job["stage"] not in ("done", "rejected"):
            seg.append(f"[bold cyan]▸ {a}[/]")
        elif a in acted:
            seg.append(f"[green]✓ {a}[/]")
        else:
            seg.append(f"[dim]{a}[/]")
    rail = "  →  ".join(seg)
    if job["stage"] == "done":
        rail += f"   ⇒   [{'bold green' if job['result'] == 'PASS' else 'bold red'}]{job['result']}[/]"
    logs = "\n".join(f"[dim]{s:<15}[/] [cyan]{ag:<13}[/] {m}" for (s, ag, m) in trace[-12:]) or "[dim]starting…[/]"
    return Panel(Group(Text.from_markup(rail), Text(""), Text.from_markup(logs)),
                 title=f"[b]autoresearch[/] · {job['brief'][:56]}",
                 subtitle=f"stage={job['stage']} · bounces={job.get('bounces', {})}", border_style="cyan")

async def run_brief(brief, constraints, mode="hitl"):
    job = {"brief": brief, "constraints": list(constraints), "mode": mode, "stage": "queue",
           "bounces": {}, "evidence": None, "approach": None, "plan": None, "code": None, "result": None}
    trace = []
    def log(agent, msg):
        print(f"  [{job['stage']:<16}] {agent:<14} {msg}", flush=True); trace.append((job["stage"], agent, msg))
    # --- MLflow run (DS dashboard: params + per-stage trace + verdicts + bounces + result) ---
    try:
        import mlflow
        mlflow.set_experiment("autoresearch")
        _ml = mlflow.start_run(run_name=datetime.datetime.now().strftime("ontology-%H%M%S"))
        mlflow.log_param("brief", brief[:240]); mlflow.log_param("model", MODEL)
        mlflow.log_param("mode", mode); mlflow.log_param("constraints", ", ".join(constraints) or "none")
    except Exception:
        _ml = None
    print(f"\n=== autoresearch REAL · model={MODEL} · NO MOCKS ===\nbrief: {brief}\nconstraints: {constraints}", flush=True)
    _av, _ = atelier_status(); _gb = governor_budget()
    _gov = (f"governor {_gb.get('free_budget_gb',0):.0f}/{_gb.get('budget_gb',0):.0f}GB free, parallel={_gb.get('num_parallel')}"
            if _gb else "governor offline")
    print(f"house: Linux box · sandbox=local-docker[{'up' if DockerExecutor.available() else 'DOWN'}] · "
          f"atelier-GPU @ {ATELIER_URL} [{'up' if _av else 'down'}] · {_gov}\n", flush=True)
    if _rich_ok():                                                                # live rail+log panel (interactive / AR_RICH=1)
        from rich.live import Live
        with Live(_rich_view(job, trace), refresh_per_second=8) as live:
            def rlog(agent, msg):
                trace.append((job["stage"], agent, msg)); live.update(_rich_view(job, trace))
            for _ in range(MAX_TICKS):
                if job["stage"] in ("done", "rejected") or job.get("_paused"): break   # F3
                await tick(job, rlog); live.update(_rich_view(job, trace))
    else:                                                                          # plain prints (batch / non-tty) — unchanged
        for _ in range(MAX_TICKS):
            if job["stage"] in ("done", "rejected") or job.get("_paused"): break      # F3: stop cleanly when paused at the gate
            await tick(job, log)
    print(f"\n=== FINISHED: stage={job['stage']} result={job['result']} bounces={job['bounces']} ===", flush=True)
    summary_path = write_summary(job, trace)
    print(f"  summary -> {os.path.relpath(summary_path, os.path.dirname(os.path.abspath(__file__)))}", flush=True)
    if _ml is not None:
        try:
            import mlflow
            mlflow.log_metric("result_code", {"PASS": 1, "FAIL": 0, "ERROR": -1}.get(job["result"], -2))
            mlflow.log_metric("stages", len(trace))
            if job.get("run_exit") is not None:
                mlflow.log_metric("docker_exit", job["run_exit"])
            for k, v in job["bounces"].items():
                mlflow.log_metric(f"bounce_{k}", v)
            mlflow.log_param("final_stage", job["stage"]); mlflow.log_param("result", str(job["result"]))
            mlflow.set_tag("shadow", "1" if SHADOW else "0")          # DSPy filter: shadow rows = critic verdict vs runner truth
            _verds = {}
            for _st, _ag, _m in trace:
                if _ag in ("taste", "reviewer", "code-reviewer"):
                    _verds[_ag] = str(_m).split()[0][:20]              # last verdict per critic
            for _ag, _v in _verds.items():
                mlflow.log_param("verdict_" + _ag.replace("-", "_"), _v)
            # full DS-traceable artifacts: pitch, plan, code, verdicts/feedback, docker stdout/stderr + errors
            for key, fname in [("evidence", "01_evidence.md"), ("approach", "02_approach_pitch.md"),
                               ("plan", "03_plan.md"), ("code", "04_code.py"),
                               ("last_review", "05_reviewer_feedback.md"), ("last_coderev", "06_code_review.md"),
                               ("run_output", "07_docker_run_output.txt")]:
                if job.get(key):
                    mlflow.log_text(str(job[key]), fname)
            mlflow.log_dict({"trace": [{"i": i, "stage": s, "agent": a, "verdict_or_msg": m}
                                       for i, (s, a, m) in enumerate(trace)]}, "run.json")
            mlflow.log_text(open(summary_path).read(), "summary.md")
            mlflow.end_run()
        except Exception:
            pass
    return job

if __name__ == "__main__":
    import sys
    brief = os.environ.get("AR_BRIEF") or (sys.argv[1] if len(sys.argv) > 1 else
            "Automatically generate an ontology (entities, attributes, relationships) from tabular data")
    cons = [c.strip() for c in os.environ.get("AR_CONSTRAINTS", "").split(",") if c.strip()]
    asyncio.run(run_brief(brief, cons, mode="hitl"))
