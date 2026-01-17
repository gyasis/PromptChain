"""Integration tests for TUI workflow display (T092).

These tests verify that workflow state is properly displayed in the
TUI status bar with progress indicators and visual updates.
"""

import pytest


class TestWorkflowTUIDisplay:
    """Test workflow display in TUI status bar."""

    @pytest.fixture
    def status_bar(self):
        """Create StatusBar instance."""
        from promptchain.cli.tui.status_bar import StatusBar
        return StatusBar()

    def test_no_workflow_displays_empty(self, status_bar):
        """Integration: Status bar without workflow shows no workflow info.

        Given: StatusBar with no workflow set
        When: render() is called
        Then: Output does not contain "Workflow:"
        """
        status_bar.update_session_info(
            session_name="test",
            active_agent="default"
        )

        rendered = status_bar.render()

        assert "Workflow:" not in rendered
        assert "test" in rendered
        assert "default" in rendered

    def test_workflow_with_0_percent_progress(self, status_bar):
        """Integration: Workflow at 0% shows starting indicator.

        Given: Workflow with 0% progress
        When: Status bar is rendered
        Then: Shows cyan starting icon (○) and 0%
        """
        status_bar.update_session_info(
            workflow_objective="Build authentication system",
            workflow_progress=0.0
        )

        rendered = status_bar.render()

        assert "Workflow:" in rendered
        assert "Build authentication system" in rendered
        assert "0%" in rendered
        # Cyan indicator for starting (not checking exact emoji due to Rich formatting)

    def test_workflow_with_50_percent_progress(self, status_bar):
        """Integration: Workflow at 50% shows in-progress indicator.

        Given: Workflow with 50% progress
        When: Status bar is rendered
        Then: Shows yellow half-filled icon (◐) and 50%
        """
        status_bar.update_session_info(
            workflow_objective="Refactor database layer",
            workflow_progress=50.0
        )

        rendered = status_bar.render()

        assert "Workflow:" in rendered
        assert "Refactor database layer" in rendered
        assert "50%" in rendered

    def test_workflow_with_100_percent_progress(self, status_bar):
        """Integration: Workflow at 100% shows completion indicator.

        Given: Workflow with 100% progress
        When: Status bar is rendered
        Then: Shows green completion icon (✓) and 100%
        """
        status_bar.update_session_info(
            workflow_objective="Complete testing suite",
            workflow_progress=100.0
        )

        rendered = status_bar.render()

        assert "Workflow:" in rendered
        assert "Complete testing suite" in rendered
        assert "100%" in rendered

    def test_workflow_long_objective_truncated(self, status_bar):
        """Integration: Long workflow objectives are truncated with ellipsis.

        Given: Workflow with objective >40 chars
        When: Status bar is rendered
        Then: Objective is truncated to 40 chars + "..."
        """
        long_objective = "This is a very long workflow objective that exceeds the maximum display length"
        status_bar.update_session_info(
            workflow_objective=long_objective,
            workflow_progress=25.0
        )

        rendered = status_bar.render()

        assert "Workflow:" in rendered
        assert "..." in rendered
        # Full objective should not be in rendered output
        assert long_objective not in rendered
        # First part should be present
        assert "This is a very long workflow" in rendered

    def test_workflow_update_changes_display(self, status_bar):
        """Integration: Updating workflow state updates display.

        Flow:
        1. Set initial workflow at 25%
        2. Update to 75%
        3. Verify display reflects new state
        """
        # Initial state
        status_bar.update_session_info(
            workflow_objective="Deploy application",
            workflow_progress=25.0
        )
        rendered1 = status_bar.render()
        assert "25%" in rendered1

        # Update progress
        status_bar.update_session_info(
            workflow_progress=75.0
        )
        rendered2 = status_bar.render()
        assert "75%" in rendered2
        assert "Deploy application" in rendered2

    def test_workflow_cleared_hides_display(self, status_bar):
        """Integration: Clearing workflow removes display.

        Flow:
        1. Set active workflow
        2. Clear workflow by setting empty objective
        3. Verify workflow no longer shown
        """
        # Set workflow
        status_bar.update_session_info(
            workflow_objective="Initial task",
            workflow_progress=50.0
        )
        rendered1 = status_bar.render()
        assert "Workflow:" in rendered1

        # Clear workflow
        status_bar.update_session_info(
            workflow_objective="",
            workflow_progress=0.0
        )
        rendered2 = status_bar.render()
        assert "Workflow:" not in rendered2

    def test_workflow_with_other_status_elements(self, status_bar):
        """Integration: Workflow displays alongside other status elements.

        Given: Status bar with session, agent, tokens, and workflow
        When: render() is called
        Then: All elements are present and separated by |
        """
        status_bar.update_session_info(
            session_name="test-session",
            active_agent="coder",
            model_name="gpt-4.1-mini-2025-04-14",
            message_count=10,
            token_count=2000,
            max_tokens=4000,
            workflow_objective="Code review workflow",
            workflow_progress=60.0
        )

        rendered = status_bar.render()

        # Check all elements present
        assert "Session: test-session" in rendered or "test-session" in rendered
        assert "coder" in rendered
        assert "gpt-4.1-mini-2025-04-14" in rendered or "gpt-4" in rendered
        assert "Messages: 10" in rendered
        assert "Tokens:" in rendered
        assert "Workflow:" in rendered
        assert "Code review workflow" in rendered
        assert "60%" in rendered

        # Check separator
        assert "|" in rendered

    def test_workflow_progress_percentage_formatting(self, status_bar):
        """Integration: Progress percentage formatted without decimals.

        Given: Workflow with fractional progress (e.g., 33.333%)
        When: Status bar is rendered
        Then: Progress shown as whole number (33%)
        """
        status_bar.update_session_info(
            workflow_objective="Test task",
            workflow_progress=33.333
        )

        rendered = status_bar.render()

        assert "33%" in rendered
        # Should not show decimals
        assert "33.3" not in rendered

    def test_workflow_reactive_property_triggers_rerender(self, status_bar):
        """Integration: Changing workflow properties triggers reactive update.

        Given: StatusBar with reactive workflow properties
        When: workflow_progress is changed
        Then: render() reflects new value immediately
        """
        status_bar.workflow_objective = "Reactive test"
        status_bar.workflow_progress = 10.0

        rendered1 = status_bar.render()
        assert "10%" in rendered1

        # Change via property (not update_session_info)
        status_bar.workflow_progress = 90.0

        rendered2 = status_bar.render()
        assert "90%" in rendered2
        assert "Reactive test" in rendered2


class TestWorkflowDisplayEdgeCases:
    """Test edge cases for workflow display."""

    @pytest.fixture
    def status_bar(self):
        """Create StatusBar instance."""
        from promptchain.cli.tui.status_bar import StatusBar
        return StatusBar()

    def test_workflow_with_exactly_40_chars(self, status_bar):
        """Edge: Objective exactly 40 chars not truncated.

        Given: Objective with exactly 40 characters
        When: Status bar is rendered
        Then: Full objective shown without ellipsis
        """
        objective_40 = "A" * 40  # Exactly 40 chars
        status_bar.update_session_info(
            workflow_objective=objective_40,
            workflow_progress=50.0
        )

        rendered = status_bar.render()

        assert objective_40 in rendered
        assert "..." not in rendered

    def test_workflow_with_41_chars(self, status_bar):
        """Edge: Objective 41 chars truncated to 40.

        Given: Objective with 41 characters
        When: Status bar is rendered
        Then: Truncated to 37 chars + "..."
        """
        objective_41 = "A" * 41
        status_bar.update_session_info(
            workflow_objective=objective_41,
            workflow_progress=50.0
        )

        rendered = status_bar.render()

        assert "..." in rendered
        # Full objective should not appear
        assert objective_41 not in rendered

    def test_workflow_with_negative_progress(self, status_bar):
        """Edge: Negative progress treated as 0%.

        Given: Workflow with negative progress value
        When: Status bar is rendered
        Then: Shows starting indicator (cyan ○)
        """
        status_bar.update_session_info(
            workflow_objective="Edge case test",
            workflow_progress=-10.0
        )

        rendered = status_bar.render()

        # Negative progress should still render without error
        assert "Workflow:" in rendered
        assert "Edge case test" in rendered

    def test_workflow_with_progress_over_100(self, status_bar):
        """Edge: Progress >100% shown as completion.

        Given: Workflow with progress >100%
        When: Status bar is rendered
        Then: Shows green completion icon (✓)
        """
        status_bar.update_session_info(
            workflow_objective="Over-complete",
            workflow_progress=150.0
        )

        rendered = status_bar.render()

        assert "Workflow:" in rendered
        assert "150%" in rendered  # Shows actual value
        # Should use green completion color

    def test_workflow_with_none_values(self, status_bar):
        """Edge: None values ignored gracefully.

        Given: update_session_info called with None workflow values
        When: Status bar is rendered
        Then: Previous values retained
        """
        # Set initial values
        status_bar.update_session_info(
            workflow_objective="Initial",
            workflow_progress=50.0
        )

        # Update with None (should not change values)
        status_bar.update_session_info(
            workflow_objective=None,
            workflow_progress=None
        )

        rendered = status_bar.render()

        # Original values should remain
        assert "Initial" in rendered
        assert "50%" in rendered

    def test_workflow_empty_string_objective(self, status_bar):
        """Edge: Empty string objective hides workflow.

        Given: Workflow objective set to empty string
        When: Status bar is rendered
        Then: No workflow display shown
        """
        status_bar.update_session_info(
            workflow_objective="",
            workflow_progress=50.0
        )

        rendered = status_bar.render()

        # Empty objective should hide entire workflow display
        assert "Workflow:" not in rendered
