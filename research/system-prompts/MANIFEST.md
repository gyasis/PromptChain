# Collected System Prompts — Manifest

Third-party reference material for the PromptChain TUI foundation-prompt research.
**Source:** `github.com/x1xhlol/system-prompts-and-models-of-ai-tools` (leaked-system-prompt inventory).
Collected 2026-06-27. These are reference-only; not PromptChain code.

**Base for assembly:** `anthropic/claude-sonnet-4.6.txt` (the designated structural base).

| Local file | Source path (in inventory repo) | Size | Notes |
|---|---|---|---|
| `anthropic/claude-sonnet-4.6.txt` | `Anthropic/Claude Sonnet 4.6.txt` | 98,989 B | **BASE** — newest Claude model prompt, largest |
| `anthropic/claude-code-2.0.txt` | `Anthropic/Claude Code 2.0.txt` | 57,324 B | Claude Code agent (closest peer to our TUI) |
| `cursor/agent-prompt-2.0.txt` | `Cursor Prompts/Agent Prompt 2.0.txt` | 38,844 B | Cursor coding agent |
| `devin/prompt.txt` | `Devin AI/Prompt.txt` | 34,714 B | Autonomous coding agent |
| `v0/prompt.txt` | `v0 Prompts and Tools/Prompt.txt` | 46,186 B | Vercel v0 (UI/code gen) |
| `cline/prompt.txt` | `Open Source prompts/Cline/Prompt.txt` | 47,083 B | Open-source coding agent |
| `roocode/prompt.txt` | `Open Source prompts/RooCode/Prompt.txt` | 43,961 B | Open-source coding agent |
| `codex-cli/prompt.txt` | `Open Source prompts/Codex CLI/openai-codex-cli-system-prompt-20250820.txt` | 23,864 B | OpenAI Codex CLI |
| `gemini-cli/prompt.txt` | `Open Source prompts/Gemini CLI/google-gemini-cli-system-prompt.txt` | 18,976 B | Google Gemini CLI |
| `warp/prompt.txt` | `Warp.dev/Prompt.txt` | 14,595 B | Warp terminal agent |
| `augment/claude-4-sonnet.txt` | `Augment Code/claude-4-sonnet-agent-prompts.txt` | 10,836 B | Augment (Claude variant) |
| `augment/gpt-5.txt` | `Augment Code/gpt-5-agent-prompts.txt` | 15,035 B | Augment (GPT-5 variant) |
| `windsurf/prompt-wave-11.txt` | `Windsurf/Prompt Wave 11.txt` | 11,699 B | Windsurf Cascade |
| `amp/claude-4-sonnet.yaml` | `Amp/claude-4-sonnet.yaml` | 66,230 B | Amp (prompt + tools, YAML) |

| `opencode/` (prompt.txt + 8 variants + routing.ts) | `github.com/sst/opencode` (MIT, source) | — | **Model-agnostic** harness; per-family prompt variants + `default.txt` core |
| `goose/` (system.md, tiny_model_system.md, subagent_system.md, toolshim_inject.txt, ancillary) | `github.com/block/goose` (source) | — | **Model-agnostic** harness; tiered core/tiny + Toolshim tool-call fallback |

Total: 16 sources (14 leaked + opencode + Goose — the last two are open-source, model-agnostic, our
closest precedents). Secondary source if needed: `github.com/jujumilk3/leaked-system-prompts`. See
`architecture/04-opencode-goose-lessons.md` for the model-agnostic mining.
