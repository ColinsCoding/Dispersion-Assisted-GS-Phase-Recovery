# Jupyter Notebook Cleanup Checklist for Submission
## Dispersion-Assisted GS Phase Recovery — Summer 2026

---

## Overview: 3 Main Notebooks to Polish

### 1. **phase_retrieval.ipynb** (Main deliverable)
- Status: ✅ Exists, has structure
- Action items:
  - [ ] Add executive summary section (1-2 cells at top)
  - [ ] Verify all plots render correctly
  - [ ] Check for broken imports (phycv, pandas, etc.)
  - [ ] Add cross-references to other notebooks
  - [ ] Ensure output persistence (all cells executed)
  - [ ] Add "How to run this notebook" cell at top

### 2. **dispersion_gs_gallery.ipynb** (Visualization & theory)
- Status: ✅ Exists
- Action items:
  - [ ] Add descriptive intro to each visualization
  - [ ] Verify 3D plots render in all environments
  - [ ] Add interpretation text below each plot
  - [ ] Link to phase_retrieval.ipynb

### 3. **dispersion_gs_gradient_descent.ipynb** (Alternative algorithm)
- Status: ✅ Exists
- Action items:
  - [ ] Add comparison section vs GS algorithm
  - [ ] Verify PyTorch integration works
  - [ ] Add convergence analysis plots
  - [ ] Note when to use this vs standard GS

---

## New Notebooks to Create (Tie together)

### 4. **README_NOTEBOOKS.md** (Navigation guide)
- What: Index of all notebooks, reading order, dependencies
- Why: Submission reviewers need clear structure
- Time: 30 min

### 5. **SUBMISSION_CHECKLIST.md** (Final validation)
- What: Reproduction steps, dependencies, expected outputs
- Why: Proves everything works end-to-end
- Time: 20 min

---

## Detailed Actions

### Phase 1: Assessment (10 min)
1. Open each notebook in Jupyter
2. Run all cells, capture errors
3. Note missing imports, broken paths
4. Identify missing explanatory text

### Phase 2: Repairs (20 min per notebook)
1. Fix import errors (update paths, add fallbacks)
2. Verify all plots save and display
3. Add docstring cells where needed

### Phase 3: Documentation (15 min)
1. Add inter-notebook links
2. Create README_NOTEBOOKS.md
3. Add "run order" instructions

### Phase 4: Validation (10 min)
1. Execute full pipeline once
2. Verify all figures save
3. Create SUBMISSION_CHECKLIST.md

**Total time: ~90 minutes**

---

## Execution Plan

**Best approach:** Use Jupyter `nbconvert` + custom Python script to:
1. Execute all notebooks and check for errors
2. Generate HTML reports
3. Verify all plots exist

**Script to create:** `validate_notebooks.py`
- Runs each notebook with timeout
- Captures errors
- Generates summary

---

## Success Criteria

✅ All notebooks run without errors  
✅ All plots display and save  
✅ Cross-references work  
✅ Clear running instructions  
✅ Ready for GitHub release  

---

## Files to Modify/Create

```
notebooks/
├── phase_retrieval.ipynb                    [MODIFY - add intro + verify]
├── dispersion_gs_gallery.ipynb             [MODIFY - add explanations]
├── dispersion_gs_gradient_descent.ipynb    [MODIFY - add comparisons]
├── griffiths_dispersion_bridge.ipynb       [NEW - already created ✓]
├── README_NOTEBOOKS.md                     [NEW - create guide]
├── SUBMISSION_CHECKLIST.md                 [NEW - validation steps]
└── validate_notebooks.py                   [NEW - test runner]
```

---

Generated: 2026-06-16
