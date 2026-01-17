COMMAND_EXIT_CODE=$?

# Save new state
echo "$(pwd)" > "/tmp/promptchain_session_8fa256a4-def4-4413-84cc-6edda58669cc/working_dir"
env | grep -E '^[A-Z_][A-Z0-9_]*=' > "/tmp/promptchain_session_8fa256a4-def4-4413-84cc-6edda58669cc/env_vars" || true

exit $COMMAND_EXIT_CODE
