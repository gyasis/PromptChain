# Complete Guide: SuperClaude Framework & Claude Code SubAgents

## Part 1: SuperClaude Framework Installation & Usage

### Prerequisites

Before installing SuperClaude Framework, ensure you have:

- **Claude Code CLI** installed (`npm install -g @anthropic-ai/claude-code`)
- **Node.js 18+** with npm 10.x+
- **Python 3.8+** (for MCP server integrations)
- **Git** configured with user credentials

### Installation Methods

#### Method 1: Using pipx (Recommended)

```bash
# Install pipx if you don't have it
brew install pipx  # macOS
# OR
python3 -m pip install --user pipx  # Linux/Windows
python3 -m pipx ensurepath

# Install SuperClaude
pipx install superclaude

# Run the installer to configure Claude Code
superclaude install
```

#### Method 2: Direct Installation from Git

```bash
# Clone the repository
git clone https://github.com/SuperClaude-Org/SuperClaude_Framework.git
cd SuperClaude_Framework

# Run the installer script
./install.sh
```

#### Method 3: Using uv (Modern Python Package Manager)

```bash
# Install uv
curl -Ls https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install SuperClaude
uv pip install SuperClaude

# Run installer
python3 -m SuperClaude install
```

#### Method 4: From PyPI with uv

```bash
uv add SuperClaude
python3 -m SuperClaude install
```

### Installation Options

After installing the package, you can customize the installation:

```bash
# Interactive installation (select components)
python3 -m SuperClaude install --interactive

# Minimal installation (core framework only)
python3 -m SuperClaude install --minimal

# Developer profile installation
python3 -m SuperClaude install --profile developer
```

### Post-Installation

1. **Restart Claude Code** to load the new commands
2. **Verify installation**:
   ```bash
   superclaude install --list
   superclaude doctor
   ```

### Configuration Files

After installation, SuperClaude creates configuration files:

- `~/.claude/settings.json` - Main configuration
- `~/.claude/*.md` - Framework behavior files
- `~/.claude/commands/` - Custom commands directory

### Using SuperClaude Framework

SuperClaude provides:

1. **16 Specialized Commands** for common development tasks
2. **9 Pre-defined Personas**:
   - Architect
   - Frontend Specialist
   - Backend Expert
   - DevOps Engineer
   - Security Auditor
   - Performance Optimizer
   - Documentation Specialist
   - Test Engineer
   - Code Reviewer

3. **Enhanced MCP Server Integrations**
4. **Team Workflow Templates**

### Example Usage

```bash
# Start Claude Code
claude

# Use SuperClaude commands (they appear automatically)
# Example: Use the architect persona
> Design a microservices architecture for an e-commerce platform

# SuperClaude automatically routes to the appropriate persona
```

---

## Part 2: Claude Code SubAgents Installation & Usage

### Understanding SubAgents

SubAgents are specialized AI assistants that:
- Operate in isolated context windows
- Have dedicated expertise (defined by their prompt)
- Use custom tool permissions
- Focus on specific objectives

### Creating SubAgents

#### Method 1: Interactive Creation (Easiest)

```bash
# Start Claude Code
claude

# Access the agents interface
/agents

# Follow the prompts:
# 1. Select "Create New Agent"
# 2. Choose scope:
#    - Project-level (`.claude/agents/`) - Only for current project
#    - User-level (`~/.claude/agents/`) - Available across all projects
# 3. Enter agent name and description
# 4. Define specialized knowledge
# 5. Select allowed tools (Read, Edit, Grep, Glob, Bash, etc.)
# 6. Set when to invoke automatically
```

#### Method 2: File-Based Creation (More Control)

Create a markdown file in the appropriate directory:

**For Project-Level Agent:**
```bash
# Create directory if it doesn't exist
mkdir -p .claude/agents

# Create agent file
touch .claude/agents/production-validator.md
```

**For User-Level Agent:**
```bash
# Create directory if it doesn't exist
mkdir -p ~/.claude/agents

# Create agent file
touch ~/.claude/agents/frontend-developer.md
```

### Agent File Format

Edit the agent file with this structure:

```markdown
---
name: production-validator
description: Automatically reviews code for production readiness.
tools: Read, Grep, Glob
permissionMode: manual  # Options: manual, acceptEdits, acceptAll
model: sonnet  # Optional: specify model
---

You are my production code quality enforcer. Here's exactly what I care about:

**IMMEDIATE BLOCKERS:**
- TODO comments or FIXME notes
- Hardcoded API keys, passwords, or tokens
- Console.log statements
- Debug breakpoints
- Test-only code in production paths

**CODE QUALITY:**
- Proper error handling
- Input validation
- Security best practices
- Performance considerations

**REVIEW PROCESS:**
1. Scan for immediate blockers
2. Check code quality standards
3. Provide actionable feedback
4. Suggest improvements

When reviewing code, be thorough but concise. Focus on issues that could cause production problems.
```

### Example SubAgent Templates

#### Frontend Developer Agent

```markdown
---
name: frontend-developer
description: Build modern, responsive frontends using React, Vue, or vanilla JS.
tools: Read, Edit, Grep
permissionMode: acceptEdits
---

You are a frontend development specialist focused on:

**Core Competencies:**
- Component-based architecture
- Performance optimization
- Responsive design
- Accessibility (WCAG 2.1)
- Modern CSS (Grid, Flexbox, CSS Variables)

**Best Practices:**
- Use semantic HTML
- Optimize bundle sizes
- Implement lazy loading
- Follow mobile-first approach
- Write accessible code

**Tech Stack Preferences:**
- React with TypeScript
- Tailwind CSS for styling
- React Query for data fetching
- Vitest for testing
```

#### Backend Developer Agent

```markdown
---
name: backend-developer
description: Develop robust backend systems with APIs, focusing on scalability and security.
tools: Read, Edit, Grep, Bash
permissionMode: acceptEdits
---

You are a backend development specialist focused on:

**Technical Expertise:**
- API development (REST, GraphQL)
- Database optimization
- Security best practices
- Authentication & authorization
- Caching strategies

**Architecture Patterns:**
- Microservices
- Event-driven architecture
- Repository pattern
- Dependency injection

**Security Focus:**
- Input validation
- SQL injection prevention
- XSS protection
- Rate limiting
- Secure authentication
```

### Using SubAgents

#### Automatic Invocation

Claude Code automatically selects the appropriate subagent based on your prompt:

```bash
claude

# Claude Code automatically routes to frontend-developer agent
> Create a responsive navigation component with React

# Claude Code automatically routes to backend-developer agent
> Design a REST API endpoint for user authentication
```

#### Manual Invocation

You can explicitly invoke a specific agent:

```bash
# In Claude Code
> @frontend-developer Create a login form component

# Or use the agent name directly
> Use the production-validator agent to review src/api/auth.js
```

### Managing SubAgents

#### List All Agents

```bash
# In Claude Code
/agents list

# Or check the directory
ls ~/.claude/agents/  # User-level
ls .claude/agents/    # Project-level
```

#### Edit an Agent

```bash
# Edit the agent file directly
nano ~/.claude/agents/frontend-developer.md

# Or in Claude Code
/agents edit frontend-developer
```

#### Delete an Agent

```bash
# Delete the agent file
rm ~/.claude/agents/agent-name.md

# Or in Claude Code
/agents delete agent-name
```

---

## Part 3: Transferring Agents Between Systems

### Understanding Agent Storage Locations

Agents are stored in two locations:

1. **Project-Level Agents**: `.claude/agents/` (in your project directory)
   - Only available in that specific project
   - Version-controlled with your project
   - Shared with team members via Git

2. **User-Level Agents**: `~/.claude/agents/` (in your home directory)
   - Available across all projects
   - Not version-controlled by default
   - Personal to your machine

### Method 1: Manual File Transfer

#### Step 1: Locate Your Agent Files

```bash
# Find project-level agents
find . -path "*/.claude/agents/*.md" -type f

# Find user-level agents
ls ~/.claude/agents/
```

#### Step 2: Copy Agent Files

**For Project-Level Agents:**
```bash
# On source system
cd /path/to/project
cp -r .claude/agents/ /path/to/backup/agents/

# On destination system
cd /path/to/project
cp -r /path/to/backup/agents/ .claude/agents/
```

**For User-Level Agents:**
```bash
# On source system
cp -r ~/.claude/agents/ ~/backup-agents/

# Transfer to destination system (via USB, network, etc.)
# Then on destination system:
cp -r ~/backup-agents/ ~/.claude/agents/
```

### Method 2: Using Git (Recommended for Project-Level Agents)

Project-level agents are automatically included in your Git repository:

```bash
# Ensure .claude/agents/ is tracked
git add .claude/agents/
git commit -m "Add project agents"
git push

# On destination system
git pull
# Agents are now available
```

### Method 3: Automated Transfer Script (Recommended)

Use the included script in the `agent-transfer/` folder for easy export and import:

**Quick Usage:**
```bash
# Navigate to the agent-transfer folder
cd agent-transfer

# Export agents (creates timestamped backup)
./agent-transfer.sh export

# Export with custom filename
./agent-transfer.sh export my-agents-backup.tar.gz

# Import agents (auto-detects Claude Code)
./agent-transfer.sh import claude-agents-backup_20250101_120000.tar.gz
```

**Features:**
- Automatically finds user-level and project-level agents
- Detects Claude Code installation on import
- Handles conflicts with existing agents
- Preserves agent metadata
- Works across different systems

**Manual Script (Alternative):**

If you prefer to create your own script:

**Export Script (`export-agents.sh`):**
```bash
#!/bin/bash

# Export user-level agents
mkdir -p ~/claude-agents-backup
cp -r ~/.claude/agents/ ~/claude-agents-backup/user-agents/

# Export project-level agents (if in current directory)
if [ -d ".claude/agents" ]; then
    cp -r .claude/agents/ ~/claude-agents-backup/project-agents/
fi

# Create a tarball
tar -czf ~/claude-agents-backup.tar.gz -C ~ claude-agents-backup/

echo "Agents exported to ~/claude-agents-backup.tar.gz"
```

**Import Script (`import-agents.sh`):**
```bash
#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: ./import-agents.sh /path/to/claude-agents-backup.tar.gz"
    exit 1
fi

# Extract the backup
tar -xzf "$1" -C ~/

# Import user-level agents
if [ -d ~/claude-agents-backup/user-agents ]; then
    mkdir -p ~/.claude/agents/
    cp -r ~/claude-agents-backup/user-agents/* ~/.claude/agents/
    echo "User-level agents imported"
fi

# Import project-level agents (if .claude directory exists)
if [ -d ~/claude-agents-backup/project-agents ] && [ -d ".claude" ]; then
    mkdir -p .claude/agents/
    cp -r ~/claude-agents-backup/project-agents/* .claude/agents/
    echo "Project-level agents imported"
fi

# Cleanup
rm -rf ~/claude-agents-backup

echo "Agents imported successfully!"
```

**Usage:**
```bash
# Make scripts executable
chmod +x export-agents.sh import-agents.sh

# Export on source system
./export-agents.sh

# Transfer the tarball to destination system
# Then import
./import-agents.sh ~/claude-agents-backup.tar.gz
```

### Method 4: Using Claude Code Templates (Community Tool)

The Claude Code Templates tool can help manage and share agents:

```bash
# Install Claude Code Templates
npx claude-code-templates@latest

# Export agents
# (Check the tool's documentation for export features)

# Import agents from the community
npx claude-code-templates@latest --agent=development-team/frontend-developer
```

### Method 5: Manual Copy-Paste

For a single agent:

1. **On Source System:**
   ```bash
   # View the agent file
   cat ~/.claude/agents/my-agent.md
   ```

2. **Copy the entire content** (including YAML frontmatter)

3. **On Destination System:**
   ```bash
   # Create the directory if needed
   mkdir -p ~/.claude/agents/
   
   # Create the agent file
   nano ~/.claude/agents/my-agent.md
   
   # Paste the content and save
   ```

### Verifying Agent Transfer

After transferring agents:

```bash
# List agents in Claude Code
claude
/agents list

# Test an agent
> @my-agent-name Test this agent
```

### Best Practices for Agent Transfer

1. **Version Control Project Agents**: Always commit project-level agents to Git
   ```bash
   git add .claude/agents/
   git commit -m "Add project agents"
   ```

2. **Document Agent Dependencies**: Note any MCP servers or tools required
   ```markdown
   ---
   name: my-agent
   description: My custom agent
   tools: Read, Edit
   required_mcps: github-mcp, filesystem-mcp
   ---
   ```

3. **Test After Transfer**: Verify agents work correctly on the new system
   ```bash
   claude
   /agents list
   > @agent-name Test functionality
   ```

4. **Backup Regularly**: Keep backups of your user-level agents
   ```bash
   # Add to your backup script
   cp -r ~/.claude/agents/ ~/backups/claude-agents-$(date +%Y%m%d)/
   ```

5. **Share via Repository**: Create a shared agents repository for your team
   ```bash
   # Create a team agents repo
   git clone https://github.com/your-org/claude-agents.git
   cd claude-agents
   
   # Add your agent
   cp ~/.claude/agents/my-agent.md ./agents/
   git add agents/my-agent.md
   git commit -m "Add my-agent"
   git push
   
   # Team members can clone and symlink
   ln -s ~/claude-agents/agents/* ~/.claude/agents/
   ```

### Troubleshooting Agent Transfer

**Issue: Agent not appearing after transfer**

```bash
# Check file permissions
chmod 644 ~/.claude/agents/*.md

# Verify YAML frontmatter is correct
head -10 ~/.claude/agents/my-agent.md

# Restart Claude Code
# Exit and restart
claude
```

**Issue: Agent tools not working**

- Verify required MCP servers are installed on the new system
- Check tool permissions in the agent file
- Ensure Claude Code has access to the tools

**Issue: Agent context not loading**

- Check that the agent file has proper markdown formatting
- Verify the YAML frontmatter is valid
- Ensure the description field is not empty

---

## Quick Reference

### SuperClaude Framework
- **Install**: `pipx install superclaude && superclaude install`
- **Verify**: `superclaude doctor`
- **Config**: `~/.claude/settings.json`

### Claude Code SubAgents
- **Create**: `/agents` in Claude Code
- **Location (Project)**: `.claude/agents/`
- **Location (User)**: `~/.claude/agents/`
- **Format**: Markdown with YAML frontmatter

### Agent Transfer
- **Project Agents**: Use Git (automatically version-controlled)
- **User Agents**: Copy `~/.claude/agents/` directory
- **Backup**: Regular backups recommended

---

## Additional Resources

- [SuperClaude Framework GitHub](https://github.com/SuperClaude-Org/SuperClaude_Framework)
- [Claude Code Documentation](https://docs.anthropic.com/claude-code)
- [Claude Code SubAgents Guide](https://docs.anthropic.com/en/docs/claude-code/sub-agents)
- [17 SubAgent Templates](https://github.com/Njengah/claude-code-cheat-sheet/blob/main/subagents.md)
- [Claude Code Templates](https://aitmpl.com)

