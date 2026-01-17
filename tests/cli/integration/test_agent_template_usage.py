"""Integration tests for agent template instantiation and usage (T098).

Tests end-to-end workflow of creating agents from templates and using them
in sessions with SessionManager integration.
"""

import pytest
from pathlib import Path

from promptchain.cli.models.agent_config import Agent, HistoryConfig
from promptchain.cli.session_manager import SessionManager
from promptchain.cli.utils.agent_templates import create_from_template


@pytest.fixture
def temp_sessions_dir(tmp_path):
    """Create temporary sessions directory."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return sessions_dir


@pytest.fixture
def session_manager(temp_sessions_dir):
    """Create session manager with temp directory."""
    return SessionManager(sessions_dir=temp_sessions_dir)


# ============================================================================
# T098: Template Instantiation and Immediate Usage
# ============================================================================


class TestTemplateInstantiation:
    """Test suite for template creation and usage (T098)."""

    def test_create_researcher_agent_and_add_to_session(self, session_manager):
        """Create researcher agent from template and add to session."""
        # Create session
        session = session_manager.create_session(
            name="research-project",
            working_directory=Path.cwd()
        )

        # Create agent from template
        agent = create_from_template("researcher", "my-researcher")

        # Add to session
        session.agents[agent.name] = agent
        session.active_agent = agent.name
        session_manager.save_session(session)

        # Verify agent is persisted correctly
        loaded_session = session_manager.load_session(session.id)
        assert "my-researcher" in loaded_session.agents
        loaded_agent = loaded_session.agents["my-researcher"]

        # Verify agent properties preserved
        assert loaded_agent.model_name == "openai/gpt-4"
        assert loaded_agent.history_config.max_tokens == 8000
        assert loaded_agent.history_config.max_entries == 50
        assert len(loaded_agent.instruction_chain) == 3
        assert "web_search" in loaded_agent.tools

    def test_create_coder_agent_and_add_to_session(self, session_manager):
        """Create coder agent from template and add to session."""
        session = session_manager.create_session(
            name="coding-project",
            working_directory=Path.cwd()
        )

        # Create agent from template
        agent = create_from_template("coder", "python-dev")

        # Add to session
        session.agents[agent.name] = agent
        session.active_agent = agent.name
        session_manager.save_session(session)

        # Verify agent is persisted correctly
        loaded_session = session_manager.load_session(session.id)
        assert "python-dev" in loaded_session.agents
        loaded_agent = loaded_session.agents["python-dev"]

        # Verify agent properties preserved
        assert loaded_agent.model_name == "openai/gpt-4"
        assert loaded_agent.history_config.max_tokens == 4000
        assert loaded_agent.history_config.max_entries == 20
        assert len(loaded_agent.instruction_chain) == 3
        assert "execute_code" in loaded_agent.tools

    def test_create_analyst_agent_and_add_to_session(self, session_manager):
        """Create analyst agent from template and add to session."""
        session = session_manager.create_session(
            name="analysis-project",
            working_directory=Path.cwd()
        )

        # Create agent from template
        agent = create_from_template("analyst", "data-analyst")

        # Add to session
        session.agents[agent.name] = agent
        session.active_agent = agent.name
        session_manager.save_session(session)

        # Verify agent is persisted correctly
        loaded_session = session_manager.load_session(session.id)
        assert "data-analyst" in loaded_session.agents
        loaded_agent = loaded_session.agents["data-analyst"]

        # Verify agent properties preserved
        assert loaded_agent.model_name == "openai/gpt-4"
        assert loaded_agent.history_config.max_tokens == 8000
        assert loaded_agent.history_config.max_entries == 50
        assert len(loaded_agent.instruction_chain) == 4
        assert "data_analysis" in loaded_agent.tools

    def test_create_terminal_agent_and_add_to_session(self, session_manager):
        """Create terminal agent from template and add to session."""
        session = session_manager.create_session(
            name="terminal-project",
            working_directory=Path.cwd()
        )

        # Create agent from template
        agent = create_from_template("terminal", "bash-exec")

        # Add to session
        session.agents[agent.name] = agent
        session.active_agent = agent.name
        session_manager.save_session(session)

        # Verify agent is persisted correctly
        loaded_session = session_manager.load_session(session.id)
        assert "bash-exec" in loaded_session.agents
        loaded_agent = loaded_session.agents["bash-exec"]

        # Verify agent properties preserved
        assert loaded_agent.model_name == "openai/gpt-3.5-turbo"
        assert loaded_agent.history_config.enabled is False
        assert loaded_agent.history_config.max_tokens == 0
        assert len(loaded_agent.instruction_chain) == 1
        assert "execute_shell" in loaded_agent.tools

    def test_create_multiple_agents_from_templates_in_session(self, session_manager):
        """Create multiple agents from different templates in same session."""
        session = session_manager.create_session(
            name="multi-agent-project",
            working_directory=Path.cwd()
        )

        # Create multiple agents from templates
        researcher = create_from_template("researcher", "researcher-1")
        coder = create_from_template("coder", "coder-1")
        analyst = create_from_template("analyst", "analyst-1")
        terminal = create_from_template("terminal", "terminal-1")

        # Add all to session
        session.agents["researcher-1"] = researcher
        session.agents["coder-1"] = coder
        session.agents["analyst-1"] = analyst
        session.agents["terminal-1"] = terminal
        session.active_agent = "researcher-1"
        session_manager.save_session(session)

        # Verify all agents persisted correctly (note: default agent is auto-created)
        loaded_session = session_manager.load_session(session.id)
        assert "researcher-1" in loaded_session.agents
        assert "coder-1" in loaded_session.agents
        assert "analyst-1" in loaded_session.agents
        assert "terminal-1" in loaded_session.agents

        # Verify each agent has correct properties
        assert loaded_session.agents["researcher-1"].history_config.max_tokens == 8000
        assert loaded_session.agents["coder-1"].history_config.max_tokens == 4000
        assert loaded_session.agents["analyst-1"].history_config.max_tokens == 8000
        assert loaded_session.agents["terminal-1"].history_config.enabled is False

    def test_create_agent_with_model_override_and_add_to_session(self, session_manager):
        """Create agent with custom model override and verify persistence."""
        session = session_manager.create_session(
            name="custom-model-project",
            working_directory=Path.cwd()
        )

        # Create agent with model override
        agent = create_from_template(
            "researcher",
            "claude-researcher",
            model_override="anthropic/claude-3-opus-20240229"
        )

        # Add to session
        session.agents[agent.name] = agent
        session_manager.save_session(session)

        # Verify custom model persisted
        loaded_session = session_manager.load_session(session.id)
        loaded_agent = loaded_session.agents["claude-researcher"]
        assert loaded_agent.model_name == "anthropic/claude-3-opus-20240229"

        # Other template properties should remain
        assert loaded_agent.history_config.max_tokens == 8000

    def test_create_agent_with_description_override_and_add_to_session(self, session_manager):
        """Create agent with custom description override and verify persistence."""
        session = session_manager.create_session(
            name="custom-desc-project",
            working_directory=Path.cwd()
        )

        # Create agent with description override
        custom_desc = "ML research specialist for deep learning papers"
        agent = create_from_template(
            "researcher",
            "ml-researcher",
            description_override=custom_desc
        )

        # Add to session
        session.agents[agent.name] = agent
        session_manager.save_session(session)

        # Verify custom description persisted
        loaded_session = session_manager.load_session(session.id)
        loaded_agent = loaded_session.agents["ml-researcher"]
        assert loaded_agent.description == custom_desc

    def test_template_agent_usage_count_updates(self, session_manager):
        """Verify template agent usage statistics update correctly."""
        session = session_manager.create_session(
            name="usage-tracking-project",
            working_directory=Path.cwd()
        )

        # Create agent from template
        agent = create_from_template("researcher", "test-researcher")
        session.agents[agent.name] = agent
        session_manager.save_session(session)

        # Simulate agent usage
        agent.update_usage()
        session_manager.save_session(session)

        # Verify usage count updated
        loaded_session = session_manager.load_session(session.id)
        loaded_agent = loaded_session.agents["test-researcher"]
        assert loaded_agent.usage_count == 1
        assert loaded_agent.last_used is not None

    def test_template_agent_has_template_metadata(self, session_manager):
        """Verify template agents have template metadata in agent.metadata."""
        session = session_manager.create_session(
            name="metadata-project",
            working_directory=Path.cwd()
        )

        # Create agents from different templates
        researcher = create_from_template("researcher", "researcher-1")
        terminal = create_from_template("terminal", "terminal-1")

        session.agents["researcher-1"] = researcher
        session.agents["terminal-1"] = terminal
        session_manager.save_session(session)

        # Verify template metadata preserved
        loaded_session = session_manager.load_session(session.id)

        researcher_meta = loaded_session.agents["researcher-1"].metadata
        assert researcher_meta["template"] == "researcher"
        assert researcher_meta["template_metadata"]["category"] == "research"
        assert researcher_meta["template_metadata"]["complexity"] == "high"

        terminal_meta = loaded_session.agents["terminal-1"].metadata
        assert terminal_meta["template"] == "terminal"
        assert terminal_meta["template_metadata"]["category"] == "execution"
        assert terminal_meta["template_metadata"]["response_time"] == "fast"

    def test_template_agent_instruction_chain_independence(self, session_manager):
        """Verify multiple agents from same template have independent instruction chains."""
        session = session_manager.create_session(
            name="independence-project",
            working_directory=Path.cwd()
        )

        # Create two agents from same template
        researcher1 = create_from_template("researcher", "researcher-1")
        researcher2 = create_from_template("researcher", "researcher-2")

        # Modify first agent's instruction chain
        researcher1.instruction_chain.append("Additional custom step")

        session.agents["researcher-1"] = researcher1
        session.agents["researcher-2"] = researcher2
        session_manager.save_session(session)

        # Verify second agent's chain was not affected
        loaded_session = session_manager.load_session(session.id)
        agent1_chain = loaded_session.agents["researcher-1"].instruction_chain
        agent2_chain = loaded_session.agents["researcher-2"].instruction_chain

        assert len(agent1_chain) == 4  # 3 original + 1 custom
        assert len(agent2_chain) == 3  # 3 original only

    def test_terminal_agent_is_recognized_as_terminal_agent(self, session_manager):
        """Verify terminal agents are recognized via is_terminal_agent property."""
        session = session_manager.create_session(
            name="terminal-check-project",
            working_directory=Path.cwd()
        )

        # Create terminal and non-terminal agents
        terminal = create_from_template("terminal", "bash-exec")
        researcher = create_from_template("researcher", "researcher-1")

        session.agents["bash-exec"] = terminal
        session.agents["researcher-1"] = researcher
        session_manager.save_session(session)

        # Verify terminal agent detection
        loaded_session = session_manager.load_session(session.id)
        assert loaded_session.agents["bash-exec"].is_terminal_agent is True
        assert loaded_session.agents["researcher-1"].is_terminal_agent is False

    def test_template_agents_can_be_deleted_from_session(self, session_manager):
        """Verify template agents can be removed from sessions."""
        session = session_manager.create_session(
            name="deletion-project",
            working_directory=Path.cwd()
        )

        # Create and add agent
        agent = create_from_template("researcher", "temp-researcher")
        session.agents["temp-researcher"] = agent
        session_manager.save_session(session)

        # Delete agent
        del session.agents["temp-researcher"]
        if session.active_agent == "temp-researcher":
            # SessionManager falls back to "default" when active_agent becomes invalid
            session.active_agent = "default"
        session_manager.save_session(session)

        # Verify deletion persisted
        loaded_session = session_manager.load_session(session.id)
        assert "temp-researcher" not in loaded_session.agents
        # SessionManager defaults to "default" agent when active_agent is deleted
        assert loaded_session.active_agent == "default"

    def test_template_agents_survive_session_reload(self, session_manager):
        """Verify template agents survive multiple save/load cycles."""
        session = session_manager.create_session(
            name="persistence-project",
            working_directory=Path.cwd()
        )

        # Create agent from template
        agent = create_from_template("coder", "persistent-coder")
        session.agents["persistent-coder"] = agent
        session_manager.save_session(session)

        # Load and re-save multiple times
        for _ in range(3):
            loaded_session = session_manager.load_session(session.id)
            session_manager.save_session(loaded_session)

        # Verify agent properties still intact
        final_session = session_manager.load_session(session.id)
        final_agent = final_session.agents["persistent-coder"]

        assert final_agent.name == "persistent-coder"
        assert final_agent.model_name == "openai/gpt-4"
        assert final_agent.history_config.max_tokens == 4000
        assert len(final_agent.instruction_chain) == 3
        assert final_agent.metadata["template"] == "coder"
