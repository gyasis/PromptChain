# PromptChain CLI Demos

This directory contains comprehensive demonstration materials for the PromptChain CLI, showcasing all Phase 6-9 features in realistic workflows.

## Available Demos

### Product Analysis Demo (Main Demo)

**Scenario**: Two-day product research and analysis workflow

**Files**:
- `DEMO_GUIDE.md` - Complete setup and running instructions
- `DEMO_TRANSCRIPT.md` - Full annotated transcript with examples
- `product_analysis_demo.sh` - Executable demo script
- `demo_config.yml` - Example configuration export
- `output/` - Generated demo artifacts

**Duration**: 15-20 minutes (10 min per day)

**Features Demonstrated**:
- ✅ Phase 6: Token Optimization (35% savings)
- ✅ Phase 7: Workflow State Management
- ✅ Phase 8: Agent Templates (researcher, terminal, analyst)
- ✅ Phase 9: Polish Features (config export, stats)

## Quick Start

### 1. Prerequisites

```bash
# Install PromptChain
cd /home/gyasis/Documents/code/PromptChain
pip install -e .

# Set up environment
cat > .env << EOF
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here  # Optional
EOF
```

### 2. Run Demo

**Automated**:
```bash
./demos/product_analysis_demo.sh
```

**Manual (Interactive)**:
```bash
# Open guide and transcript
cat demos/DEMO_GUIDE.md
cat demos/DEMO_TRANSCRIPT.md

# Start interactive session
promptchain --session demo-session

# Follow commands from transcript
```

### 3. Review Artifacts

```bash
# View generated configuration
cat demos/output/demo_config.yml

# Check token statistics
cat demos/output/history_stats_*.txt

# Inspect workflow state
cat demos/output/workflow_state.json
```

## Demo Files Structure

```
demos/
├── README.md                          # This file
├── DEMO_GUIDE.md                      # Setup & running instructions
├── DEMO_TRANSCRIPT.md                 # Complete annotated transcript
├── product_analysis_demo.sh           # Executable demo script
├── demo_config.yml                    # Example configuration export
└── output/                            # Generated artifacts (created on run)
    ├── day1_commands.txt              # Day 1 command sequence
    ├── day2_commands.txt              # Day 2 command sequence
    ├── demo_config.yml                # Configuration export
    ├── history_stats_day1.txt         # Day 1 token statistics
    ├── history_stats_day2.txt         # Day 2 token statistics
    └── workflow_state.json            # Workflow state snapshot
```

## Customization

### Change Demo Scenario

Edit `product_analysis_demo.sh`:

```bash
# Line 10-15: Demo configuration
PRODUCT_CATEGORY="smart home devices"  # Change category
COMPETITORS="Google, Amazon, Apple"    # Change competitors
RESEARCH_FOCUS="pricing strategies"    # Change focus area
```

### Adjust Token Limits

Edit `demo_config.yml`:

```yaml
agents:
  - name: market-researcher
    history_config:
      max_tokens: 8000  # Increase for more context
      max_entries: 50   # Adjust history depth
```

### Add Custom Agent Templates

Edit `demo_config.yml`:

```yaml
agent_templates:
  custom_agent:
    base_model: openai/gpt-4
    temperature: 0.5
    system_role: Custom Specialist
    history_config:
      max_tokens: 6000
      max_entries: 30
```

## Presentation Tips

### For Live Demos

1. **Pre-run**: Execute demo once to verify timing and outputs
2. **Add pauses**: Modify script to add `read -p "Press Enter..."` commands
3. **Highlight features**: Point out token savings, workflow states
4. **Show artifacts**: Display generated files in separate terminal

### For Recorded Demos

1. **Record session**: Use `asciinema rec demo.cast`
2. **Add annotations**: Edit recording with time markers
3. **Speed up**: Increase playback speed for repetitive parts
4. **Add voiceover**: Explain features as they're demonstrated

### For Documentation

1. **Capture screenshots**: Key moments from transcript
2. **Export logs**: Include JSONL logs for deep analysis
3. **Annotate code**: Add inline comments explaining steps

## Troubleshooting

### Script Fails to Run

```bash
# Check syntax
bash -n demos/product_analysis_demo.sh

# Run with verbose output
bash -x demos/product_analysis_demo.sh

# Check permissions
chmod +x demos/product_analysis_demo.sh
```

### API Key Issues

```bash
# Verify .env file exists
ls -la .env

# Check key format
cat .env | grep API_KEY

# Test API access
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('OpenAI key:', os.getenv('OPENAI_API_KEY')[:10])"
```

### Session Directory Issues

```bash
# Create sessions directory
mkdir -p ~/.promptchain/sessions

# Check permissions
ls -ld ~/.promptchain/sessions

# Clear old sessions
rm -rf ~/.promptchain/sessions/demo-*
```

## Advanced Usage

### Multi-Model Comparison

Run demo with different models:

```bash
# GPT-4
DEMO_MODEL=openai/gpt-4 ./demos/product_analysis_demo.sh

# Claude
DEMO_MODEL=anthropic/claude-3-sonnet-20240229 ./demos/product_analysis_demo.sh

# Compare outputs
diff demos/output/gpt4_report.md demos/output/claude_report.md
```

### Token Usage Analysis

Extract and analyze token metrics:

```bash
# Parse statistics
grep "Total tokens" demos/output/history_stats_*.txt

# Calculate savings
python3 << 'EOF'
day1_optimized = 5847
day1_baseline = 8847
savings = (day1_baseline - day1_optimized) / day1_baseline * 100
print(f"Day 1 Savings: {savings:.1f}%")
EOF
```

### Export for Team Sharing

```bash
# Create demo package
tar czf promptchain_demo.tar.gz demos/

# Share configuration
cp demos/demo_config.yml ~/team_configs/

# Export session data
# (Run after interactive demo)
# /session export demo-complete.json
```

## Creating New Demos

### Demo Template Structure

```bash
demos/
├── <demo-name>/
│   ├── README.md              # Demo-specific instructions
│   ├── TRANSCRIPT.md          # Annotated command transcript
│   ├── <demo-name>.sh         # Executable script
│   ├── config.yml             # Configuration export
│   └── output/                # Generated artifacts
```

### Demo Checklist

- [ ] Clear scenario and objective
- [ ] Realistic workflow (15-20 minutes)
- [ ] Demonstrates multiple features
- [ ] Includes token optimization
- [ ] Shows workflow state management
- [ ] Uses agent templates
- [ ] Exports configuration
- [ ] Generates useful artifacts
- [ ] Fully documented with annotations
- [ ] Tested end-to-end

### Demo Best Practices

1. **Keep it realistic**: Use actual use cases, not contrived examples
2. **Show progression**: Build complexity gradually across demo
3. **Highlight savings**: Make token optimization benefits clear
4. **Include annotations**: Explain WHY, not just WHAT
5. **Provide artifacts**: Generate shareable outputs
6. **Test thoroughly**: Run demo multiple times before sharing
7. **Document well**: Include setup, troubleshooting, customization

## FAQ

**Q: Can I run demos without API keys?**
A: No, demos require real API access. For offline demos, use recorded sessions or mock responses.

**Q: How much do demos cost to run?**
A: Product Analysis demo costs ~$0.40-0.60 in API fees (with optimization). Without optimization, ~$0.80-1.00.

**Q: Can I use local models (Ollama)?**
A: Yes, modify `demo_config.yml` to use `ollama/llama2` or similar models.

**Q: How do I reset a demo?**
A: Delete session data:
```bash
rm -rf ~/.promptchain/sessions/demo-*
rm -rf demos/output/*
```

**Q: Can I contribute new demos?**
A: Yes! Follow the demo template structure and submit a PR.

## Support

- **Documentation**: See `DEMO_GUIDE.md` for detailed instructions
- **GitHub Issues**: Report bugs or request features
- **Community**: Share your demo scenarios with other users

## Version History

- **v1.0.0** (2025-11-24): Initial product analysis demo
  - Phase 6-9 feature coverage
  - Multi-day workflow demonstration
  - Comprehensive documentation

---

**Last Updated**: 2025-11-24
**PromptChain Version**: 0.4.2+
**Maintainer**: PromptChain Development Team
