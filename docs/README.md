# PromptChain documentation

This page lists the current docs by topic. Finished plans, phase summaries and one-off reports live in [archive/](archive/README.md). Nobody maintains them.

PromptChain has no default model. Name the model in your code, or set per-role models in `~/.promptchain/models.yaml` (see [Models](#models)).

## Start here

- [Project README](../README.md): overview and install.
- [Installation](../INSTALLATION.md), [Quickstart](../QUICKSTART.md), [Usage](../USAGE.md): the agentic chat team.
- [Quick start (library)](guides/QUICK_START.md) and [multi-step quickstart](promptchain_quickstart_multistep.md).
- [Framework guide](promptchain_guide.md): chains, prompt engineering, execution.
- [CLI quick reference](CLI_QUICK_REFERENCE.md).

## For LLM agents

- [PromptChain for LLMs](llms/PROMPTCHAIN_FOR_LLMS.md): the canonical reference, with the anti-pattern list.
- [llms.txt](llms/llms.txt): index of the reference and the recipes.
- [Recipes](llms/recipes/): tested copy-paste patterns (basic chain, agentic step, router, MCP, models, cascade, trace export, and more).
- [Feedback log](llms/FEEDBACK_LOG.md) and [issue drafts](llms/issues/).
- The bundled skill lives at `promptchain/data/skills/promptchain.md`. Install it with `promptchain install-skill`.

## Models

- [Models and tool calling](llms/recipes/recipe-models-and-tool-calling.md): model strings, tool-calling support, the models file (`promptchain models show|init`).
- [Per-step models](per_step_models.md).
- [Two-tier routing](TWO_TIER_ROUTING_GUIDE.md).
- [Model router](llms/recipes/recipe-model-router.md): route by measured ability. The profiler also probes the governor llamacpp lane.
- [Model management](model_management.md): VRAM management for local providers.
- [Hugging Face models](huggingface_integration.md).

## Chains and steps

- [Chain injection examples](chain_injection_examples.md), [dynamic chain examples](dynamic_chain_examples.md), [head-tail technique](head_tail_technique.md), [web search tutorial](web_search_tutorial.md).
- [Chainbreaker guide](chainbreaker_guide.md), [circular runs](circular_runs.md), [chain of draft](chain_of_draft.md).
- [Registering functions](registering_functions_to_chain.md), [PrePrompt](preprompt_usage.md), [Prompt Engineer](prompt_engineer.md).
- [ExternalLoop](external_loop.md) ([infographic](external_loop_infographic.html)).
- [ExecutionHistoryManager](ExecutionHistoryManager.md), [log suppression](log_suppression.md).
- [Validity statistics reference](validity_stats_reference.md).

## Agents and orchestration

- [AgenticStepProcessor](agentic_step_processor.md) and [its best practices](agentic_step_processor_best_practices.md).
- [AgentChain](utils/agent_chain.md), [StateAgent](utils/state_agent.md), [agent templates](agent-templates.md).
- [OrchestratorSupervisor](milestones/ORCHESTRATOR_SUPERVISOR_ARCHITECTURE.md).
- [Blackboard architecture](BLACKBOARD_ARCHITECTURE.md).
- [Safety features](SAFETY_FEATURES.md).
- [Simple agent user chat](simple_agent_user_chat.md).
- Multi-agent references: [message bus](multiagent/MESSAGE_BUS_QUICK_REFERENCE.md), [mental models](multiagent/MENTAL_MODELS_QUICK_REFERENCE.md), [imports](multiagent/MULTIAGENT_IMPORT_QUICK_REFERENCE.md), [tasks and blackboard commands](multiagent/TASKS_BLACKBOARD_QUICK_REFERENCE.md).
- [Architecture vision](architecture/ARCHITECTURE_VISION.md) and the [durable, verified TUI loop proposal](architecture/tui_agent_durable_verified_loop.md) (design drafts; check the code before you rely on them).

## MCP

- [MCP tool integration](MCP_server_use.md), [MCP setup and prompts](mcp_setup_and_prompts.md), [MCP Tool Hijacker](mcp_tool_hijacker.md).
- [Installing MCP servers from GitHub](MCP_GITHUB_INSTALLATION_GUIDE.md) and the [Elementor MCP reference](ELEMENTOR_MCP_TOOLS_REFERENCE.md) cover third-party servers.

## Observability

- [Observability system](observability/README.md): events, callbacks, metadata. Also [event system](observability/event-system.md), [MCP events](observability/mcp-events.md), [public APIs](observability/public-apis.md), [best practices](observability/best-practices.md), [migration guide](observability/migration-guide.md), [plugin architecture](observability/OBSERVABILITY_PLUGIN_ARCHITECTURE.md).
- [MLflow observability guide](observability_guide.md), [MLflow observer usage](mlflow_observer_usage.md), [MLflow cheatsheet](mlflow/MLflow_Cheatsheet.md).

## CLI and TUI

- [Activity logging quick start](cli/ACTIVITY_LOGGING_QUICK_START.md).
- [Error handling guide](error-handling-guide.md) and [error message catalog](T105_ERROR_MESSAGE_CATALOG.md).
- [Config export and import](T111_CONFIG_EXPORT_IMPORT_GUIDE.md) with [an example export](example_config_export.yml) and [config show output](config_show_example.txt).
- [`/task` command design](agent_comms_tui_task_command.md).
- [Lookout bridge](lookout_bridge.md).
- Tool APIs: [file operations](api/file-operations-tools.md), [terminal tool visual debugging](api/terminal-tool-visual-debugging.md).

## Retrieval and patterns

- [LightRAG pattern adapters](patterns/README.md): [branching](patterns/branching.md), [hybrid search fusion](patterns/hybrid_search_fusion.md), [multi-hop](patterns/lightrag_multi_hop.md), [speculative execution](patterns/lightrag_speculative_execution.md), [sharded](patterns/sharded.md), [mixture of strategies](patterns/mixture_of_strategies.md).
- [LightRAG detailed review](LIGHTRAG_DEEP_DIVE.md), [PostgreSQL setup for LightRAG](POSTGRESQL_LIGHTRAG_SETUP.md), [LightRAG and RAG-Anything overview](rag/LIGHTRAG_AND_RAG_ANYTHING_OVERVIEW.md).
- [OKF import](okf_import.md), [DeepLake backup and sync](deeplake_backup_and_sync_guide.md).

## Background and reference

- [DSPy 3.0 guide](dspy/DSPy_3.0_Comprehensive_Guide.md) and [custom evaluations](dspy/DSPy_3.0_Custom_Evaluations_Guide.md): third-party tutorials.
- [Claude Code Router guide](claude-code-router-guide.md): third-party tool.
- [True agent autonomy](research/true_agent_autonomy.md): research brief.
- [Adversarial analysis of the enhanced AgenticStepProcessor](ADVERSARIAL_ANALYSIS_SUMMARY.md) and the [Codex harness gap analysis](codex-harness-gap-analysis.html): kept because specs and PRDs cite them.
- [Releasing](RELEASING.md), [CHANGELOG](../CHANGELOG.md), [references](../REFERENCES.md), [security](../SECURITY.md).
- [Examples](../examples/TUTORIAL_TESTING_GUIDE.md): tutorial testing guide.
