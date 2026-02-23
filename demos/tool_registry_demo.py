#!/usr/bin/env python3
"""
Tool Registry System Demonstration

This script demonstrates the key features of the PromptChain CLI Tool Registry:
- Decorator-based tool registration
- Parameter validation
- Tool discovery and lookup
- OpenAI function calling integration
- Error handling

Run: python demos/tool_registry_demo.py
"""

import json
from promptchain.cli.tools import (
    registry,
    ToolCategory,
    ParameterSchema,
    ToolRegistrationError,
    ToolValidationError,
    ToolNotFoundError
)


def demo_section(title: str):
    """Print section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def demo_registration():
    """Demonstrate tool registration."""
    demo_section("1. Tool Registration")

    @registry.register(
        category=ToolCategory.FILESYSTEM,
        description="Read file contents with specified encoding",
        parameters={
            "path": {
                "type": "string",
                "required": True,
                "description": "File path to read"
            },
            "encoding": {
                "type": "string",
                "default": "utf-8",
                "description": "File encoding",
                "enum": ["utf-8", "ascii", "latin-1"]
            }
        },
        tags=["io", "files", "read"],
        examples=[
            "fs_read(path='config.json')",
            "fs_read(path='data.txt', encoding='latin-1')"
        ]
    )
    def fs_read(path: str, encoding: str = "utf-8") -> str:
        """Read file contents."""
        try:
            with open(path, encoding=encoding) as f:
                return f.read()
        except FileNotFoundError:
            return f"Error: File '{path}' not found"
        except Exception as e:
            return f"Error reading file: {e}"

    @registry.register(
        category=ToolCategory.SHELL,
        description="Execute shell command and return output",
        parameters={
            "command": {
                "type": "string",
                "required": True,
                "description": "Shell command to execute"
            },
            "timeout": {
                "type": "integer",
                "default": 30,
                "description": "Timeout in seconds"
            }
        },
        tags=["exec", "shell"]
    )
    def shell_exec(command: str, timeout: int = 30) -> str:
        """Execute shell command."""
        return f"[Demo] Would execute: {command} (timeout: {timeout}s)"

    @registry.register(
        category=ToolCategory.ANALYSIS,
        description="Analyze text and return statistics",
        parameters={
            "text": {"type": "string", "required": True, "description": "Text to analyze"},
            "include_words": {"type": "boolean", "default": False, "description": "Include word count"}
        }
    )
    def text_analyze(text: str, include_words: bool = False) -> str:
        """Analyze text statistics."""
        stats = {
            "characters": len(text),
            "lines": len(text.split("\n"))
        }
        if include_words:
            stats["words"] = len(text.split())
        return json.dumps(stats, indent=2)

    print(f"✅ Registered {len(registry.list_tools())} tools")
    print(f"   Tools: {', '.join(registry.list_tools())}")


def demo_lookup():
    """Demonstrate tool lookup methods."""
    demo_section("2. Tool Lookup & Discovery")

    # By name
    print("📌 Lookup by name:")
    tool = registry.get("fs_read")
    print(f"   Tool: {tool.name}")
    print(f"   Category: {tool.category.value}")
    print(f"   Description: {tool.description}")
    print(f"   Required params: {tool.get_required_parameters()}")
    print(f"   Optional params: {tool.get_optional_parameters()}")

    # By category
    print("\n📂 Lookup by category (FILESYSTEM):")
    fs_tools = registry.get_by_category(ToolCategory.FILESYSTEM)
    for t in fs_tools:
        print(f"   - {t.name}: {t.description}")

    # By tags
    print("\n🏷️  Lookup by tags (match any: 'io', 'exec'):")
    tagged_tools = registry.get_by_tags(["io", "exec"], match_all=False)
    for t in tagged_tools:
        print(f"   - {t.name} [tags: {', '.join(t.tags)}]")

    # List all
    print(f"\n📋 Categories in use: {', '.join(registry.list_categories())}")
    print(f"📋 Tags in use: {', '.join(registry.list_tags())}")


def demo_validation():
    """Demonstrate parameter validation."""
    demo_section("3. Parameter Validation")

    tool = registry.get("fs_read")

    # Valid parameters
    print("✅ Valid parameters:")
    try:
        tool.validate_parameters({"path": "test.txt"})
        print("   {'path': 'test.txt'} - PASSED")

        tool.validate_parameters({"path": "test.txt", "encoding": "utf-8"})
        print("   {'path': 'test.txt', 'encoding': 'utf-8'} - PASSED")
    except ToolValidationError as e:
        print(f"   FAILED: {e}")

    # Invalid parameters
    print("\n❌ Invalid parameters:")

    # Missing required
    try:
        tool.validate_parameters({})
        print("   {} - PASSED (unexpected!)")
    except ToolValidationError as e:
        print(f"   {{}} - FAILED (expected): {e}")

    # Invalid enum value
    try:
        tool.validate_parameters({"path": "test.txt", "encoding": "invalid"})
        print("   {'encoding': 'invalid'} - PASSED (unexpected!)")
    except ToolValidationError as e:
        print(f"   {{'encoding': 'invalid'}} - FAILED (expected): {e}")


def demo_execution():
    """Demonstrate tool execution."""
    demo_section("4. Tool Execution")

    # Via registry
    print("🔧 Execute via registry:")
    result = registry.execute("shell_exec", command="ls -la", timeout=10)
    print(f"   Result: {result}")

    # Via tool object
    print("\n🔧 Execute via tool object:")
    tool = registry.get("text_analyze")
    result = tool(text="Hello World\nThis is a test", include_words=True)
    print(f"   Result:\n{result}")

    # Error handling
    print("\n⚠️  Error handling:")
    try:
        registry.execute("nonexistent_tool", param="value")
    except ToolNotFoundError as e:
        print(f"   ToolNotFoundError: {e}")

    try:
        tool = registry.get("fs_read")
        tool(path=123)  # Wrong type
    except ToolValidationError as e:
        print(f"   ToolValidationError: {e}")


def demo_openai_format():
    """Demonstrate OpenAI function calling format."""
    demo_section("5. OpenAI Function Calling Format")

    # Single tool schema
    tool = registry.get("fs_read")
    schema = tool.to_openai_schema()

    print("📄 Single tool schema:")
    print(json.dumps(schema, indent=2))

    # All schemas for category
    print("\n📄 All FILESYSTEM tool schemas:")
    fs_schemas = registry.get_openai_schemas(category="filesystem")
    print(f"   Found {len(fs_schemas)} schema(s)")
    for s in fs_schemas:
        print(f"   - {s['function']['name']}")


def demo_metadata():
    """Demonstrate metadata export."""
    demo_section("6. Registry Metadata Export")

    metadata = registry.export_metadata()

    print("📊 Registry statistics:")
    stats = metadata["statistics"]
    print(f"   Total tools: {stats['total_tools']}")
    print(f"   Total tags: {stats['total_tags']}")
    print(f"   Categories:")
    for cat, count in stats["categories"].items():
        print(f"      - {cat}: {count} tool(s)")

    print("\n📝 Tool metadata sample (fs_read):")
    fs_read_meta = metadata["tools"]["fs_read"]
    print(f"   Category: {fs_read_meta['category']}")
    print(f"   Description: {fs_read_meta['description']}")
    print(f"   Parameters: {list(fs_read_meta['parameters'].keys())}")
    print(f"   Tags: {fs_read_meta['tags']}")
    print(f"   Examples: {len(fs_read_meta['examples'])} example(s)")


def demo_advanced():
    """Demonstrate advanced features."""
    demo_section("7. Advanced Features")

    # Custom parameter schemas
    print("🔬 Custom parameter schemas:")

    file_path_schema = ParameterSchema(
        name="path",
        type="string",
        description="File path",
        required=True
    )

    @registry.register(
        category=ToolCategory.FILESYSTEM,
        description="Write file with custom schema",
        parameters={
            "path": file_path_schema,
            "content": {"type": "string", "required": True, "description": "Content"}
        }
    )
    def fs_write(path: str, content: str) -> str:
        return f"[Demo] Would write {len(content)} bytes to {path}"

    print(f"   ✅ Registered 'fs_write' with custom ParameterSchema")

    # Nested objects
    print("\n🔬 Nested object validation:")

    @registry.register(
        category=ToolCategory.UTILITY,
        description="Configure server with nested params",
        parameters={
            "config": {
                "type": "object",
                "required": True,
                "description": "Server configuration",
                "properties": {
                    "host": ParameterSchema(
                        name="host",
                        type="string",
                        description="Server host",
                        required=True
                    ),
                    "port": ParameterSchema(
                        name="port",
                        type="integer",
                        description="Server port",
                        default=8080
                    )
                }
            }
        }
    )
    def configure_server(config: dict) -> str:
        return f"Server configured: {config['host']}:{config.get('port', 8080)}"

    tool = registry.get("configure_server")
    result = tool(config={"host": "localhost", "port": 9000})
    print(f"   ✅ Nested validation: {result}")


def demo_cleanup():
    """Demonstrate cleanup operations."""
    demo_section("8. Cleanup & Management")

    print(f"📊 Current tools: {len(registry.list_tools())} registered")

    # Unregister one tool
    print("\n🗑️  Unregister 'shell_exec':")
    success = registry.unregister("shell_exec")
    print(f"   Result: {'Success' if success else 'Failed'}")
    print(f"   Remaining tools: {len(registry.list_tools())}")

    # Clear all (commented out to preserve demo state)
    # print("\n🗑️  Clear all tools:")
    # registry.clear()
    # print(f"   Remaining tools: {len(registry.list_tools())}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("  PromptChain CLI Tool Registry Demonstration")
    print("=" * 60)

    try:
        demo_registration()
        demo_lookup()
        demo_validation()
        demo_execution()
        demo_openai_format()
        demo_metadata()
        demo_advanced()
        demo_cleanup()

        print("\n" + "=" * 60)
        print("  Demo Complete!")
        print("=" * 60)
        print("\nFor more examples, see:")
        print("  - promptchain/cli/tools/README.md")
        print("  - tests/cli/tools/test_registry.py")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
