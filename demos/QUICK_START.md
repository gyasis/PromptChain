# PromptChain Demo - Quick Start

## 5-Minute Setup

```bash
# 1. Install PromptChain
cd /home/gyasis/Documents/code/PromptChain
pip install -e .

# 2. Set API key
echo "OPENAI_API_KEY=your_key_here" > .env

# 3. Run demo
./demos/product_analysis_demo.sh
```

## What You'll See

**Day 1 (Research Phase)**:
```bash
✓ Create researcher agent from template
✓ Create terminal agent (60% token savings)
✓ Research 5 smart home hub products
✓ Generate competitive analysis
✓ Save session state
```

**Day 2 (Analysis Phase)**:
```bash
✓ Resume previous session
✓ Create analyst agent from template
✓ Identify market gaps
✓ Generate executive report
✓ Export configuration
```

## Key Features Demonstrated

| Feature | Command | Savings |
|---------|---------|---------|
| Token Optimization | `/history stats` | 35% |
| Terminal Agent | `/agent create-from-template terminal` | 60% |
| Workflow State | `/workflow create` | Multi-day |
| Agent Templates | `/agent create-from-template` | 3 types |
| Config Export | `/config export` | Reproducible |

## Demo Files

- `DEMO_GUIDE.md` - Full setup instructions (5 pages)
- `DEMO_TRANSCRIPT.md` - Complete annotated transcript (20 pages)
- `product_analysis_demo.sh` - Automated demo script
- `demo_config.yml` - Example configuration
- `README.md` - Complete documentation

## Expected Output

```
Token Savings: 35% (7,243 tokens saved)
Duration: 39 minutes (over 2 days)
Files Created: 4 (reports, configs, notes)
Workflow Status: ✅ Completed
```

## Interactive Demo

```bash
# Start session
promptchain --session my-demo

# Follow commands from transcript
cat demos/DEMO_TRANSCRIPT.md
```

## Next Steps

1. **Review Guide**: `cat demos/DEMO_GUIDE.md`
2. **Read Transcript**: `cat demos/DEMO_TRANSCRIPT.md`
3. **Customize Demo**: Edit `product_analysis_demo.sh`
4. **Run Interactive**: Follow transcript manually

## Troubleshooting

**Command not found**:
```bash
pip install -e .
```

**No API key**:
```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

**Script fails**:
```bash
bash -x demos/product_analysis_demo.sh
```

---

**Time to complete**: 15-20 minutes
**API cost**: ~$0.40-0.60 (with optimization)
**Difficulty**: Intermediate
