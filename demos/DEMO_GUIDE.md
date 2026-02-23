# Product Analysis Demo - Setup & Running Guide

## Overview

This demo showcases all Phase 6-9 features of the PromptChain CLI through a realistic two-day product research and analysis workflow.

**Scenario**: A product manager researching competitor products and creating an analysis report.

**Time**: 15-20 minutes (10 min Day 1, 10 min Day 2)

**Features Demonstrated**:
- Phase 8: Agent Templates (researcher, terminal, analyst)
- Phase 6: Token Optimization (60% savings, history stats)
- Phase 7: Workflow State (create, save, resume)
- Phase 9: Polish Features (config management, error handling, docs)

## Prerequisites

### 1. Environment Setup

Create `.env` file with API keys:

```bash
# Required for demo
OPENAI_API_KEY=your_openai_key_here

# Optional (for multi-model demo)
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here
```

### 2. Install PromptChain

```bash
# From project root
cd /home/gyasis/Documents/code/PromptChain
pip install -e .
```

### 3. Verify Installation

```bash
promptchain --version
promptchain --help
```

## Running the Demo

### Method 1: Automated Demo Script

Run the complete demo automatically:

```bash
cd /home/gyasis/Documents/code/PromptChain
chmod +x demos/product_analysis_demo.sh
./demos/product_analysis_demo.sh
```

The script will:
1. Execute Day 1 (research phase)
2. Pause for you to review output
3. Execute Day 2 (analysis phase)
4. Generate demo artifacts in `demos/output/`

### Method 2: Manual Step-by-Step

Follow the transcript manually for interactive demo:

```bash
# Open transcript in one terminal
cat demos/DEMO_TRANSCRIPT.md

# Run commands in another terminal
cd /home/gyasis/Documents/code/PromptChain
promptchain
```

Copy/paste commands from the transcript section by section.

### Method 3: Semi-Automated with Pauses

For presentations, use the demo script with manual confirmation:

```bash
# Edit the script to add pauses
sed -i 's/# PAUSE_FOR_DEMO/read -p "Press Enter to continue..."/g' demos/product_analysis_demo.sh

# Run with pauses
./demos/product_analysis_demo.sh
```

## Demo Structure

### Day 1: Research Phase (10 minutes)

**Commands**: 15-20 commands
**Focus**:
- Agent template creation (researcher)
- Workflow setup
- Token optimization display
- Session persistence

**Key Moments**:
1. Create researcher agent from template (00:30)
2. Show template customization (01:00)
3. Execute research queries (02:00-07:00)
4. Display token statistics (08:00)
5. Save workflow state (09:00)

### Day 2: Analysis Phase (10 minutes)

**Commands**: 12-15 commands
**Focus**:
- Workflow resumption
- Multi-agent coordination
- Configuration export
- Error handling demonstration

**Key Moments**:
1. Resume previous session (00:30)
2. Create analyst agent (01:00)
3. Analyze findings (02:00-07:00)
4. Export configuration (08:00)
5. Final report generation (09:00)

## Expected Output

### Artifacts Generated

```
demos/output/
├── day1_session.json           # Day 1 session export
├── day2_session.json           # Day 2 session export
├── history_stats_day1.txt      # Token usage statistics
├── history_stats_day2.txt      # Token usage comparison
├── workflow_state.json         # Workflow state snapshot
├── demo_config.yml             # Configuration export
└── final_report.md             # Generated analysis report
```

### Token Savings Demonstration

Expected token usage (approximate):

**Without Optimization** (traditional approach):
- Day 1: ~15,000 tokens (full history for all agents)
- Day 2: ~20,000 tokens (accumulated history)
- **Total**: ~35,000 tokens

**With Optimization** (Phase 6 features):
- Day 1: ~9,000 tokens (terminal agent: -60%, researcher: full)
- Day 2: ~11,000 tokens (smart history management)
- **Total**: ~20,000 tokens
- **Savings**: ~43% overall (up to 60% for terminal operations)

## Troubleshooting

### Issue: Command Not Found

```bash
# Ensure PromptChain is installed
pip install -e .

# Verify installation
which promptchain
```

### Issue: API Key Errors

```bash
# Check .env file exists
ls -la .env

# Verify key format
cat .env | grep API_KEY
```

### Issue: Session Directory Errors

```bash
# Create sessions directory
mkdir -p ~/.promptchain/sessions

# Check permissions
ls -ld ~/.promptchain/sessions
```

### Issue: Demo Script Fails

```bash
# Run with verbose output
bash -x demos/product_analysis_demo.sh

# Check for syntax errors
bash -n demos/product_analysis_demo.sh
```

## Customization

### Change Demo Scenario

Edit variables in `product_analysis_demo.sh`:

```bash
# Line 10-15
PRODUCT_CATEGORY="smart home devices"  # Change category
COMPETITORS="Google, Amazon, Apple"    # Change competitors
RESEARCH_FOCUS="pricing strategies"    # Change focus
```

### Adjust Token Limits

Edit demo config in script:

```bash
# Line 50-55
MAX_HISTORY_TOKENS=4000  # Increase for more context
MAX_ENTRIES=20           # Adjust history entries
```

### Add Custom Agents

Create new agent template calls:

```bash
/agent create-from-template <type> <name> \
  --model <model> \
  --description "Custom agent description"
```

## Presentation Tips

### For Live Demos

1. **Pre-run once**: Execute demo beforehand to verify timing
2. **Use pauses**: Add `read` commands for Q&A opportunities
3. **Highlight features**: Point out token savings, workflow state
4. **Show artifacts**: Display generated files in separate terminal

### For Recorded Demos

1. **Use asciinema**: Record terminal session
   ```bash
   asciinema rec demo_recording.cast
   ./demos/product_analysis_demo.sh
   ```

2. **Add annotations**: Edit recording with markers
3. **Speed up**: Increase playback speed for repetitive parts

### For Documentation

1. **Capture screenshots**: Key moments in the transcript
2. **Export logs**: Include full JSONL logs for analysis
3. **Annotate code**: Add inline comments explaining each step

## Advanced Usage

### Multi-Model Comparison

Run demo with different models:

```bash
# GPT-4 only
DEMO_MODEL=openai/gpt-4 ./demos/product_analysis_demo.sh

# Claude comparison
DEMO_MODEL=anthropic/claude-3-sonnet-20240229 ./demos/product_analysis_demo.sh

# Compare outputs
diff demos/output/gpt4_report.md demos/output/claude_report.md
```

### Token Usage Analysis

Extract detailed token metrics:

```bash
# Parse history stats
grep "Total tokens" demos/output/history_stats_*.txt

# Calculate savings percentage
python -c "
day1 = 9000  # With optimization
baseline = 15000  # Without
print(f'Savings: {((baseline - day1) / baseline * 100):.1f}%')
"
```

### Workflow State Inspection

Examine workflow state JSON:

```bash
# Pretty print workflow state
cat demos/output/workflow_state.json | jq .

# Extract specific fields
jq '.current_step, .completed_steps' demos/output/workflow_state.json
```

## FAQ

**Q: Can I run the demo without API keys?**
A: No, the demo requires at least OpenAI API access. For offline demos, use mock responses or recorded sessions.

**Q: How long does the demo take?**
A: 15-20 minutes total (10 min each phase). Adjust timing by modifying research depth.

**Q: Can I use local models (Ollama)?**
A: Yes, modify the demo script to use `ollama/llama2` or similar models.

**Q: What if I want to demo only specific features?**
A: Comment out sections in the demo script or run individual command sequences from the transcript.

**Q: How do I reset the demo?**
A: Delete session data:
```bash
rm -rf ~/.promptchain/sessions/product-analysis-*
rm -rf demos/output/*
```

## Support

For issues or questions:
- GitHub Issues: https://github.com/yourusername/PromptChain/issues
- Documentation: `/home/gyasis/Documents/code/PromptChain/docs/`
- Email: support@promptchain.dev

## Next Steps

After completing the demo:

1. **Review Transcript**: Read `DEMO_TRANSCRIPT.md` for detailed annotations
2. **Explore Features**: Try modifying agent templates, workflow steps
3. **Build Custom Workflows**: Apply patterns to your own use cases
4. **Contribute**: Share your demo scenarios with the community

---

**Demo Version**: 1.0.0
**Last Updated**: 2025-11-24
**PromptChain Version**: 0.4.2+
