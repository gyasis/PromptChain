"""ExternalLoop examples (issue #8).

Run with NO API key — the default demo uses a pure-Python step so loop semantics
(guard, breaker, worklist, state-wrapping) are visible deterministically:

    python examples/external_loop_example.py

The 6-step sequential-PromptChain version (which DOES need an OpenAI key) is at the
bottom in `llm_distill_demo()` — call it explicitly if you have OPENAI_API_KEY set.
"""

import asyncio
import json
import os

from promptchain.utils.external_loop import ExternalLoop, over_worklist


# ----------------------------------------------------------------------------- #
# Demo 1 — natural exhaustion over a worklist (the state wraps the work)
# ----------------------------------------------------------------------------- #
async def worklist_demo():
    async def handler(item, state):
        state["results"][item] = item * item  # pretend this was a chain pass

    state = await over_worklist([1, 2, 3, 4, 5], handler, max_iters=100)
    print("worklist  ->", state["results"],
          "| stopped:", state["_stopped"], "| iters:", state["_iterations"])


# ----------------------------------------------------------------------------- #
# Demo 2 — the REQUIRED guard stops a step that never finishes on its own
# ----------------------------------------------------------------------------- #
async def guard_demo():
    async def step(it, state):
        state["count"] = state.get("count", 0) + 1
        return True  # never returns False — only the guard can stop this

    state = await ExternalLoop(max_iters=5).run(step)            # iteration guard
    print("max_iters ->", state["count"], "| stopped:", state["_stopped"])

    async def slow(it, state):
        await asyncio.sleep(0.02)
        state["count"] = state.get("count", 0) + 1
        return True

    state = await ExternalLoop(max_iters=10_000, max_seconds=0.05).run(slow)  # time guard
    print("max_seconds ->", state["count"], "| stopped:", state["_stopped"])


# ----------------------------------------------------------------------------- #
# Demo 3 — a custom breaker exits the loop (checked AFTER the required guard)
# ----------------------------------------------------------------------------- #
async def breaker_demo():
    def reached_target(it, state):
        return (state.get("total", 0) >= 30, "hit-target-30")

    async def step(it, state):
        state["total"] = state.get("total", 0) + it  # 1, 3, 6, 10, 15, 21, 28, 36...
        return True

    state = await ExternalLoop(max_iters=1000, breakers=(reached_target,)).run(step)
    print("breaker   ->", state["total"], "| stopped:", state["_stopped"],
          "| iters:", state["_iterations"])


# ----------------------------------------------------------------------------- #
# Demo 4 (opt-in, needs OPENAI_API_KEY) — ExternalLoop wrapping a 6-step chain
# ----------------------------------------------------------------------------- #
async def llm_distill_demo(docs):
    from promptchain import PromptChain

    def make_distill_chain():
        return PromptChain(
            models=["openai/gpt-4o-mini"] * 5,  # 5 string steps; the function step = no slot
            instructions=[
                "Extract the key points from this document as a bullet list:\n{input}",
                "Group these bullet points into 2-4 themes:\n{input}",
                "Order the themes from most to least important:\n{input}",
                "Write a short paragraph summary from these ordered themes:\n{input}",
                "Tighten that paragraph to exactly TWO sentences:\n{input}",
                lambda two: json.dumps({"summary": two.strip()}),
            ],
            store_steps=True,
        )

    state = {"queue": list(docs), "results": []}

    async def step(it, st):
        if not st["queue"]:
            return False
        out = await make_distill_chain().process_prompt_async(st["queue"].pop(0))
        st["results"].append(json.loads(out))
        return bool(st["queue"])

    return await ExternalLoop(max_iters=50, max_seconds=120).run(step, state)


async def main():
    print("=== ExternalLoop demos (no API key needed) ===")
    await worklist_demo()
    await guard_demo()
    await breaker_demo()
    if os.getenv("OPENAI_API_KEY"):
        print("=== LLM 6-step distill (OPENAI_API_KEY found) ===")
        state = await llm_distill_demo(["A long document about ...", "Another doc ..."])
        print("distilled ->", state["results"], "| stopped:", state["_stopped"])
    else:
        print("(set OPENAI_API_KEY to also run the 6-step LLM distill demo)")


if __name__ == "__main__":
    asyncio.run(main())
