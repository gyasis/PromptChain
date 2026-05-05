# GitHub Spec Kit - Cheat Sheet

## What is Spec Kit?
**Spec-Driven Development (SDD)** toolkit that transforms specifications into working code using AI agents. Focus on "what" to build before "how" to build it.

---

## Quick Install

```bash
# Install via uv package manager
uv tool install specify-cli --from git+https://github.com/github/spec-kit.git

# Initialize a new project
specify init my-project --ai copilot
# Options: --ai copilot | claude | gemini
```

---

## Core Workflow (4 Steps)

| Step | Command | What It Does |
|------|---------|--------------|
| **1. Specify** | `/speckit.specify` | Converts high-level idea → detailed functional spec with user stories |
| **2. Plan** | `/speckit.plan` | Creates technical implementation plan (architecture, data models, APIs) |
| **3. Tasks** | `/speckit.tasks` | Breaks plan into actionable task checklist |
| **4. Implement** | `/speckit.implement` | Executes tasks to generate code aligned with spec |

---

## All Slash Commands

### Core Commands (Required)
- `/speckit.constitution` - Set project principles & development guidelines
- `/speckit.specify` - Define requirements & user stories
- `/speckit.plan` - Create technical implementation plan
- `/speckit.tasks` - Generate task breakdown
- `/speckit.implement` - Execute implementation

### Optional Commands
- `/speckit.clarify` - Refine underspecified areas
- `/speckit.analyze` - Consistency & coverage analysis
- `/speckit.checklist` - Generate quality validation checklists

---

## Project Structure

```
my-project/
├── .specify/          # SDD templates & utility scripts
├── .github/           # Prompt definitions for slash commands
├── constitution.md    # Project rules & principles (AI must follow)
└── specs/
    └── 001-feature-name/  # Numbered feature folders
        ├── spec.md        # Functional specification
        ├── plan.md        # Technical plan
        └── tasks.md       # Task breakdown
```

---

## Key Concepts

### Constitution (`constitution.md`)
- **Non-negotiable project rules** (e.g., "always use TypeScript", "must include unit tests")
- AI agents must follow these for every task
- Acts as project "memory bank"

### Spec-Driven Development
- **Intent-first**: Define "what" before "how"
- **Executable specs**: Generate plans → tasks → code
- **Living documentation**: Version-controlled specs evolve with codebase

---

## Supported AI Agents

- GitHub Copilot
- Claude Code (Cursor)
- Amazon Q Developer CLI
- Gemini CLI

---

## Example Workflow

```bash
# 1. Initialize project
specify init task-manager --ai copilot

# 2. In your AI assistant, use slash commands:
/speckit.specify Build a task management app with user authentication, real-time collaboration, and mobile support.

/speckit.plan Use React with TypeScript, Node.js backend, PostgreSQL database

/speckit.tasks

/speckit.implement
```

---

## Why Use Spec Kit?

✅ **Reduces AI hallucinations** - Rigid plans + constitution keep AI on track  
✅ **Better collaboration** - Review plans before code is written  
✅ **Faster onboarding** - New devs read `specs/` to understand features  
✅ **Maintainable** - Specs are version-controlled and evolve with code  

---

## Resources

- **Repo**: https://github.com/github/spec-kit
- **Docs**: https://github.github.com/spec-kit/
- **Quickstart**: https://github.github.com/spec-kit/quickstart.html

---

## Pro Tips

1. **Start with constitution** - Define your project principles first
2. **Iterate on specs** - Use `/speckit.clarify` to refine before planning
3. **Review plans** - Check technical plans before implementation
4. **Version control specs** - Treat specs as first-class artifacts
5. **Use numbered folders** - `specs/001-feature/` keeps things organized








