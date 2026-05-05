---
id: recipe-prompt-loader
title: Prompt loader — load_prompts, get_prompt_by_name, PrePrompt, custom dirs
when: You want to keep prompt templates in files (not inline strings) and reference them by ID from chain steps.
api_version: 0.6.1+
---

# Prompt Loader

PromptChain has a file-based prompt registry. Templates live in directories you specify; chain steps reference them by ID instead of inlining a template string.

## A. Quick lookup (one-shot)

```python
from promptchain import load_prompts, get_prompt_by_name

# Bulk-load every prompt under one or more directories
prompts = load_prompts(["./prompts", "./shared_prompts"])
print(prompts["summarize_v1"])    # the raw template string

# Or fetch one by name (searches default + additional dirs)
template = get_prompt_by_name("summarize_v1", additional_dirs=["./prompts"])
```

## B. Use prompt IDs in chain instructions

Inside a `PromptChain`, an `instruction` of type `str` is normally treated as a literal template. **But** if it matches a known prompt ID (or the form `id:strategy`), `PrePrompt` resolves it automatically.

```python
from promptchain import PromptChain

chain = PromptChain(
    models=["openai/gpt-4o-mini"],
    instructions=[
        "summarize_v1",                # literal prompt ID — loaded from prompt dirs
        "extract_entities:cot",        # prompt ID with strategy modifier (e.g., chain-of-thought)
    ],
    additional_prompt_dirs=["./prompts"],
)
```

The dispatcher tries: literal template first → then prompt-ID lookup → raises if neither works.

## C. The `id:strategy` shorthand

If your `PrePrompt` registry has multiple variants of a prompt (e.g., `summarize_v1`, `summarize_v1:cot`, `summarize_v1:react`), the `:strategy` suffix selects one. The available strategies are project-defined; check what your `prompts/` directory ships.

## D. Where templates live (default search paths)

`PrePrompt` searches:
1. The package-shipped prompt directory (`promptchain/prompts/`)
2. Any path passed via `additional_prompt_dirs=[...]` to `PromptChain` or `load_prompts`

Files are typically `.txt` or `.md` named `<id>.txt` / `<id>.md` (sometimes with a YAML frontmatter for metadata; consult the actual `prompts/` folder for conventions).

## Common failures + fix

- **`KeyError: 'summarize_v1'`** — file not found. Check the file name matches the ID exactly, no extension in the call. Pass `additional_prompt_dirs=` if your prompts live outside the default location.
- **Template substitution silently doesn't happen** — the file content uses curly braces for non-`{input}` purposes (e.g., JSON examples). Either escape `{{` / `}}`, or restructure the template.
- **Treating an inline string as a prompt ID** — if your inline template contains spaces or template variables like `{input}`, the dispatcher correctly treats it as a literal. Only single-token strings get the prompt-ID lookup.

## Where to read the source-of-truth

- `promptchain/utils/prompt_loader.py` — `load_prompts`, `get_prompt_by_name`
- `promptchain/utils/preprompt.py` — `PrePrompt` resolver used internally
- `promptchain/prompts/` — package-shipped templates and the `BasePromptBuilder` Protocol (`base.py`, `dynamic.py`, `legacy_tui.py`) — note this is the v0.6.0+ Protocol-based system from spec 011, distinct from the file-based registry above
