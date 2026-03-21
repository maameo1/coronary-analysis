# PI Progress Update — Feb 28, 2026

---

## Slide 1: Overview

**Title**: PhD Progress Update — Topology-Aware Coronary Segmentation

- Paper 1 (MICCAI): Submitted Feb 26 — BCS metric + CT-FM L4
- Paper 2 (MIUA): In progress — Encoder adaptation study (deadline ~April 2)
- Paper 3 (TMI): Experiments running — Preprocessing + architecture ablation
- 19 HPC jobs running simultaneously

---

## Slide 2: MIUA — The Question

**How should we adapt a pretrained CT foundation model for topology-aware coronary segmentation?**

- CT-FM: SegResNet pretrained on 148K CT scans (Project LIGHTER)
- Fine-tune with Soft BCS loss (differentiable bifurcation connectivity)
- 5 adaptation strategies × 3 BCS weights = ablation grid
- Test on ImageCAS (n=250) + ASOCA cross-domain (n=40)

---

## Slide 3: MIUA — Ablation Grid Results (ImageCAS)

⚠️ **Numbers below from overlap=0.5 inference. Re-running all with overlap=0.75 (Job 8086). Will update.**

| Adaptation | λ=0.03 | λ=0.05 | λ=0.10 |
|---|---|---|---|
| **Frozen** (10M) | D=0.793 B=0.716 | D=0.792 B=0.722 | D=0.785 B=0.730 |
| **BitFit** (10M+14K) | — | D≈0.793 B≈0.720 | — |
| **Adapters** (10.2M) | — | D≈0.793 B≈0.720 | — |
| **Partial** (27.7M) | D=0.793 B=0.715 | D=0.792 B=0.724 | D=0.789 B=0.714 |
| **Full** (87.8M) | — | — | D=0.748 B=0.739 |

**Key message**: Frozen wins. Changing the encoder doesn't help (BitFit, adapters, partial) or actively hurts (full). CKA confirms: frozen preserves encoder features (≥0.84), full rewrites them (0.37 at bottleneck).

---

## Slide 4: MIUA — ASOCA Cross-Domain Transfer

**Problem**: ASOCA labels only major vessels. Our model correctly segments side branches → penalised for being right.

⚠️ **Zero-shot + threshold numbers from overlap=0.5. Re-running with overlap=0.75 (Job 8096). Low-shot already at 0.75.**

| Approach | Dice | BCS | HD95 | Overlap |
|---|---|---|---|---|
| Zero-shot (t=0.5) | 0.481 | 0.794 | 15.7 | 0.5 ⚠️ |
| Zero-shot (t=0.9) | 0.665 | 0.791 | 13.0 | 0.5 ⚠️ |
| **k=1 fine-tune (t=0.7)** | **0.706** | **0.752** | **10.6** | 0.75 ✓ |
| k=5 fine-tune | 0.695 | 0.695 | 26.8 | 0.75 ✓ |
| k=10 fine-tune | 0.671 | 0.751 | 46.4 | 0.75 ✓ |

**Key messages**:
- Threshold calibration alone recovers +18pp Dice at no BCS cost
- k=1 (just 2 training scans!) is best — more data overfits to ASOCA's sparse annotations
- Shallow decoder freeze (depth=2) preserves topology; full decoder fine-tuning destroys it

---

## Slide 5: MIUA — Surprise clDice Finding (NEW)

**Supervisor feedback**: "Conclusion too strong with only one topology loss — try clDice"

| Experiment | Freeze | Loss | Val Dice | Compare |
|---|---|---|---|---|
| EXP-048 | Frozen | BCS w=0.05 | 0.792 | — |
| EXP-045 | Full | BCS w=0.10 | 0.744 (overfit) | — |
| clDice-frozen | Frozen | clDice w=0.05 | **0.803** | +1.1pp vs BCS |
| clDice-full | Full | clDice w=0.10 | **0.808** | vs 0.744 with BCS! |
| BCS+clDice | Frozen | Both w=0.05 | Running... | — |

**CAUTION — clDice has over-promised before**:
- In MICCAI experiments, L2 (clDice) had high val Dice but test Dice was **0.787** — actually **lower** than L1 (0.790, no topology loss)
- BCS (L4, EXP-044) test Dice = 0.793 — genuinely improved over L1
- clDice may inflate val Dice without translating to test performance
- **Must run test inference on these before drawing conclusions**

**If test numbers DO hold**:
- Full unfreeze works with clDice but fails with BCS — overfitting is loss-specific
- Combined BCS+clDice experiment running — complementary signals?

**If test numbers DON'T hold (more likely based on history)**:
- Confirms that clDice inflates val metrics — BCS is the more reliable topology loss
- Strengthens the original MIUA narrative: BCS + frozen encoder is the best approach

---

## Slide 6: TMI (Paper 3) — The Question

**Does BCS loss generalise across preprocessing choices and architectures?**

- 4 preprocessing variables: HU window, patch size, spacing, augmentation
- 7 architectures: CT-FM, SwinUNETR, STU-Net, UNETR, MedNeXt, VISTA3D, nnU-Net
- All use frozen encoder + BCS w=0.05

---

## Slide 7: TMI — Preprocessing Results (DONE)

| Variable | Best Setting | Val Dice | vs Baseline (0.792) |
|---|---|---|---|
| **HU window** | [-200, 800] | **0.799** | +0.7pp |
| HU window | [-400, 1000] | 0.791 | -0.1pp |
| Patch size | 128³ | 0.794 | +0.2pp |
| Patch size | 160³ | 0.779 | -1.3pp (failed) |
| Spacing | 0.7mm | 0.789 | -0.3pp |
| Spacing | 1.5mm | 0.754 | -3.8pp (harmful) |
| **Augmentation** | Heavy | **0.796** | +0.4pp |
| Augmentation | Topo-preserving | 0.792 | same |

**Winners**: HU 800 (+0.7pp) and heavy augmentation (+0.4pp)
- Combined L1 training (HU 800 + heavy aug) submitted — job 8080

---

## Slide 8: TMI — Architecture Comparison

| Architecture | Pretrained | Val Dice | Status |
|---|---|---|---|
| **CT-FM SegResNet** | SSL 143K cardiac | **0.792** | DONE (reference) |
| **STU-Net Large** | Random (440M) | **0.801** | DONE — crossed 0.80! |
| MedNeXt-B | None (11M) | 0.798 | Running (ep46/100) |
| Aug heavy (CT-FM) | SSL 143K | 0.796 | DONE |
| SwinUNETR | Random | 0.787 | Running (ep44/50) |
| STU-Net Base | Sup TotalSeg | 0.778 | DONE |
| SwinUNETR | BTCV pretrained | 0.772 | Running (ep27/30) |
| VISTA3D | Sup 11.4K | 0.763 | DONE — disappointing |
| UNETR | Random | 0.754 | Running (ep47/50) |
| UNETR | BTCV pretrained | 0.743 | Running (ep27/30) |

**Key findings**:
- STU-Net Large (random init, 440M params) beats CT-FM — capacity matters
- VISTA3D disappointing despite massive supervised pretraining — coarse-grained pretraining doesn't help fine vessels
- Transformers (UNETR) lag behind CNNs — hierarchical attention (SwinUNETR) better than flat (UNETR)
- Domain-specific SSL (CT-FM) > general supervised (VISTA3D, STU-Net Base)

---

## Slide 9: TMI — K-Fold CV Progress

L1 baselines (folds 2-4) training well:

| Fold | Progress | Best Val Dice |
|---|---|---|
| 1 (EXP-028) | 100/100 | 0.790 (done) |
| 2 | 46/100 | **0.815** |
| 3 | 21/100 | 0.803 |
| 4 | 48/100 | **0.818** |

All folds above 0.80 — healthy. Once complete, resubmit all architecture + preprocessing experiments per fold.

---

## Slide 10: Next Steps

### MIUA (deadline ~April 2)
- [ ] **Overlap fix**: All test inference re-running with overlap=0.75 (Job 8086 ImageCAS, Job 8096 ASOCA)
- [ ] Wait for clDice + BCS+clDice results → run test inference → update paper narrative
- [ ] Ensemble + TTA experiment running (Job 8085) — likely negative result but needed for completeness
- [ ] Rerun CKA/probe analyses with final checkpoints
- [ ] Fill all paper tables with overlap=0.75 numbers
- [ ] Write up clDice comparison + nuance the conclusion

### TMI (target late 2026)
- [ ] L1 combo training (HU 800 + heavy aug, job 8080) → then L4 on top
- [ ] Re-run top architectures with best preprocessing
- [ ] Wait for L1 folds 2-4 → 4-fold CV for all experiments
- [ ] nnU-Net + BCS (separate framework)
- [ ] Test inference + bootstrap significance tests
- [ ] CKA representation analysis across architectures

### Data access (parallel)
- Ethics/data access for ACS outcome labels — has PI initiated this?

---

## Slide 11: Methodological Fix — Inference Overlap

**Issue discovered**: Inconsistent sliding window overlap across experiments.

| Script group | Old overlap | New overlap |
|---|---|---|
| ImageCAS test inference (all MIUA models) | 0.5 | **0.75** |
| ASOCA zero-shot + threshold sweep | 0.5 | **0.75** |
| ASOCA low-shot sweep | 0.75 | 0.75 (already correct) |
| Ensemble / TTA | 0.75 | 0.75 (already correct) |

**Impact**: overlap=0.75 produces smoother predictions (more averaging at patch boundaries). May slightly change all metrics. All scripts now standardised. Re-running everything before filling paper tables.

**Jobs**: 8086 (ImageCAS, 10 models × 250 cases), 8096 (ASOCA zero-shot + threshold)

---

## Slide 12: Questions / Discussion Points

1. **clDice finding**: Full unfreeze works with clDice but not BCS. Should we expand the MIUA ablation grid to include clDice at all freeze levels? Or keep it focused (frozen vs full for both losses)?

2. **MIUA narrative shift**: The original claim was "always freeze the encoder." Now it's "freeze for BCS, but clDice may not need it." Is this a stronger or weaker paper?

3. **STU-Net Large at 0.801**: It's a random-init 440M model beating CT-FM (0.792). Is this a capacity story or does it suggest CT-FM's pretraining is less valuable than we thought?

4. **ACS data access**: Timeline for ethics approval and outcome labels?
