# Tutorial Testing Guide

## ✅ Pre-Flight Validation Complete

All components verified and working:
- ✅ MLflow 3.8.1 installed in deeplake environment
- ✅ MLflow UI running on http://localhost:5000
- ✅ tutorial_helpers.py imports successfully
- ✅ PromptChain core imports working
- ✅ All mock tools functional
- ✅ All scenario builders operational
- ✅ Visualization helpers working

## 🚀 Running the Tutorial

### Step 1: Activate Environment
```bash
conda activate deeplake
```

### Step 2: Navigate to Examples Directory
```bash
cd /home/gyasis/Documents/code/PromptChain/examples
```

### Step 3: Launch Jupyter Notebook
```bash
jupyter notebook
```

### Step 4: Open Tutorial
In the Jupyter interface, open: `tutorial_full_stack_with_mlflow.ipynb`

### Step 5: Run Cells Sequentially
Execute cells one by one (Shift+Enter) in this order:

1. **Cell 1-3**: Setup and imports
   - Should see: ✅ MLflow available - full logging enabled
   - Should see: ✅ All imports successful!

2. **Cell 4-6**: Mock tools and scenarios
   - Verify mock tools work correctly

3. **Subsequent cells**: Follow the tutorial sections:
   - Section 0: Setup ✅
   - Section 1: Baseline (without improvements)
   - Section 2: Phase 1 - Two-Tier Routing
   - Section 3: Phase 2 - Blackboard Architecture
   - Section 4: Phase 3 - Safety Features
   - Section 5: Phase 4 - TAO Loop
   - Section 6: Full Stack (all features)
   - Section 7: Multi-Agent Orchestration
   - Section 8: MLflow UI Walkthrough
   - Section 9: Best Practices

### Step 6: View MLflow Results
While running cells, open http://localhost:5000 to see:
- Run metrics (tokens, costs, errors)
- Artifacts (blackboard snapshots, verification logs)
- Nested runs (multi-agent section)

## 📊 Expected Results

### Baseline Run (Section 1)
```
Tokens: ~39,334
Cost: ~$0.049
Errors: ~5
```

### Phase 1 Run (Section 2)
```
Tokens: ~39,334 (same)
Cost: ~$0.017 (65% reduction)
Errors: ~5 (same)
```

### Phase 2 Run (Section 3)
```
Tokens: ~11,125 (71.7% reduction)
Cost: ~$0.006 (compounded savings)
Errors: ~5 (same)
```

### Phase 3 Run (Section 4)
```
Tokens: ~11,125 (same as Phase 2)
Cost: ~$0.006 (same)
Errors: ~1 (80% reduction)
Dangerous ops blocked: ✅
```

### Phase 4 Run (Section 5)
```
Tokens: ~11,125 (same)
Cost: ~$0.007 (<15% overhead)
Errors: ~1 (same)
TAO phases: visible in logs
```

### Full Stack Run (Section 6)
```
Tokens: ~11,125
Cost: ~$0.007
Errors: ~1
All features: ✅
```

## 🔍 What to Check

### During Execution:
1. **No import errors** - All cells execute without ModuleNotFoundError
2. **Console output** - See progress indicators and metrics
3. **MLflow logging** - Metrics appear in UI as cells run
4. **Visualization output** - Blackboard and TAO visualizations render

### In MLflow UI:
1. **Experiment**: "promptchain_tutorial" appears
2. **Runs**: 7 runs visible (01_Baseline through 07_MultiAgent_Orchestration)
3. **Metrics**: Token counts, costs, error rates logged
4. **Artifacts**: JSON files for blackboard evolution, CoVe decisions, TAO logs
5. **Comparison**: Can select multiple runs and compare side-by-side

## 🐛 Troubleshooting

### Cell Execution Errors
**Problem**: Cell fails with error
**Solution**:
1. Check error message for missing dependencies
2. Restart kernel: Kernel → Restart
3. Run cells again from top

### MLflow Logging Issues
**Problem**: Metrics not appearing in UI
**Solution**:
1. Verify MLflow UI is running: http://localhost:5000
2. Check `MLFLOW_AVAILABLE` is True in Cell 3 output
3. Verify experiment "promptchain_tutorial" exists in UI

### Dependency Conflicts
**Problem**: Warnings about protobuf/starlette versions
**Solution**:
- These are non-critical, ignore for now
- MLflow 3.8.1 works despite warnings

### Kernel Hangs
**Problem**: Cell execution seems stuck
**Solution**:
1. Interrupt kernel: Kernel → Interrupt
2. Restart kernel: Kernel → Restart & Clear Output
3. Re-run from Cell 1

## ✅ Validation Checklist

Before committing changes, verify:
- [ ] All 30 cells execute without errors
- [ ] MLflow UI shows all 7 runs
- [ ] Metrics match expected ranges (±10%)
- [ ] Artifacts downloadable from MLflow UI
- [ ] Visualizations render correctly (Blackboard evolution, TAO logs)
- [ ] Multi-agent section creates nested runs

## 📝 Next Steps After Testing

1. **Document any issues** encountered during testing
2. **Take screenshots** of MLflow UI showing results
3. **Verify metrics** align with documented performance claims
4. **Test on fresh environment** if possible (another conda env)
5. **Commit changes** once fully validated

## 🎯 Success Criteria

✅ Tutorial runs end-to-end without manual intervention
✅ All Phase 1-4 features demonstrated successfully
✅ MLflow captures comprehensive metrics and artifacts
✅ Visualizations provide clear understanding of each phase
✅ Multi-agent orchestration shows proper agent selection

---

**Validation Script**: Run `python test_tutorial_setup.py` anytime to verify setup.

**MLflow UI**: http://localhost:5000

**Tutorial Notebook**: `tutorial_full_stack_with_mlflow.ipynb`
