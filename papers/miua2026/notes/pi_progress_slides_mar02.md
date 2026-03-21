# PI Progress Update — Mar 2, 2026 (evening)

---

## Slide 1: Overview

**Title**: PhD Progress Update — Topology-Aware Coronary Segmentation

- Paper 1 (MICCAI): Submitted Feb 26 — BCS metric + CT-FM L4
- Paper 2 (MIUA): In progress — Encoder adaptation study (deadline ~April 2)
- Paper 3 (TMI): Phase 1-2 complete, Phase 3 running — Preprocessing + architecture ablation
- HPC crash recovered (Mar 1, DNS failure) — checkpoint resumption built in
- **22 active jobs** (12 running + 10 queued) across 4 GPU nodes
- **Test inference submitted for ALL 9 TMI architectures** (Job 8139)

---

## Slide 2: MIUA — The Question

**How should we adapt a pretrained CT foundation model for topology-aware coronary segmentation?**

- CT-FM: SegResNet pretrained on 148K CT scans (Project LIGHTER)
- Fine-tune with Soft BCS loss (differentiable bifurcation connectivity)
- 5 adaptation strategies x 3 BCS weights = ablation grid
- Test on ImageCAS (n=250) + ASOCA cross-domain (n=40)

---

## Slide 3: MIUA — Ablation Grid Results (ImageCAS)

**Test metrics (overlap=0.75, n=250). L1 baseline (no BCS): D=0.792, B=0.688.**

| Adaptation | w=0.03 | w=0.05 | w=0.10 |
|---|---|---|---|
| **Frozen** (10M) | D=**0.794** B=0.705 | D=0.792 B=0.722 | D=0.785 B=**0.730** |
| **BitFit** (10M+14K) | — | D=0.793 B=0.721 | — |
| **Adapters** (10.2M) | — | (pending) | — |
| **Partial** (27.7M) | D=0.793 B=0.715 | D=0.793 B=0.719 | D=0.789 B=0.714 |
| **Full** (87.8M) | — | — | D=0.748 B=0.740 |

**Key message**: Frozen wins. Changing the encoder doesn't help (BitFit, adapters, partial) or actively hurts (full). CKA confirms: frozen preserves encoder features (>=0.84), full rewrites them (0.37 at bottleneck).

---

## Slide 4: MIUA — ASOCA Cross-Domain Transfer

**Problem**: ASOCA labels only major vessels. Our model correctly segments side branches -> penalised for being right.

**Zero-shot + threshold numbers from overlap=0.5. Re-running with overlap=0.75 (Job 8113, RUNNING). Low-shot already at 0.75.**

| Approach | Dice | BCS | HD95 | Overlap |
|---|---|---|---|---|
| Zero-shot (t=0.5) | 0.481 | 0.794 | 15.7 | 0.5 |
| Zero-shot (t=0.9) | 0.665 | 0.791 | 13.0 | 0.5 |
| **k=1 fine-tune (t=0.7)** | **0.706** | **0.752** | **10.6** | 0.75 |
| k=5 fine-tune | 0.695 | 0.695 | 26.8 | 0.75 |
| k=10 fine-tune | 0.671 | 0.751 | 46.4 | 0.75 |

**Key messages**:
- Threshold calibration alone recovers +18pp Dice at no BCS cost
- k=1 (just 2 training scans!) is best — more data overfits to ASOCA's sparse annotations
- Shallow decoder freeze (depth=2) preserves topology; full decoder fine-tuning destroys it

---

## Slide 5: MIUA — clDice Comparison (supervisor request)

**Supervisor feedback**: "Conclusion too strong with only one topology loss — try clDice"

| Experiment | Freeze | Loss | Val Dice | Status |
|---|---|---|---|---|
| EXP-048 | Frozen | BCS w=0.05 | 0.792 | Test re-running (Job 8102) |
| EXP-045 | Full | BCS w=0.10 | 0.744 (overfit) | Test re-running (Job 8102) |
| clDice-frozen | Frozen | clDice w=0.05 | **0.803** | Test inference queued (Job 8132) |
| clDice-full | Full | clDice w=0.10 | **0.808** | Test inference queued (Job 8132) |
| BCS+clDice | Frozen | Both w=0.05 | 0.799 (ep20) | Training resumed (Job 8122, RUNNING) |

**CAUTION — clDice has over-promised before**:
- In MICCAI experiments, L2 (clDice) had high val Dice but test Dice was **0.787** — actually **lower** than L1 (0.790, no topology loss)
- BCS (L4, EXP-044) test Dice = 0.793 — genuinely improved over L1
- clDice may inflate val Dice without translating to test performance
- **Test inference queued (Job 8132) — will know within 24-48h**

---

## Slide 6: TMI (Paper 3) — The Question

**Does BCS loss generalise across preprocessing choices and architectures?**

- 4 preprocessing variables: HU window, patch size, spacing, augmentation
- 9 architecture configurations: CT-FM, SwinUNETR (x2), STU-Net (x2), UNETR (x2), MedNeXt, VISTA3D
- All use frozen encoder + BCS w=0.05 (or full unfreeze for from-scratch models)

---

## Slide 7: TMI — Preprocessing Results (COMPLETE)

| Variable | Best Setting | Val Dice | vs Baseline (0.792) |
|---|---|---|---|
| **HU window** | [-200, 800] | **0.799** | +0.7pp |
| HU window | [-400, 1000] | 0.791 | -0.1pp (timed out ep19/30) |
| Patch size | 128^3 | 0.794 | +0.2pp |
| Patch size | 160^3 | 0.779 | -1.3pp (failed — never improved from L1 init) |
| Spacing | 0.7mm | 0.789 | -0.3pp (timed out ep21/30) |
| Spacing | 1.5mm | 0.754 | -3.8pp (harmful) |
| **Augmentation** | Heavy | **0.796** | +0.4pp |
| Augmentation | Topo-preserving | 0.792 | same |

**Winners**: HU 800 (+0.7pp) and heavy augmentation (+0.4pp)
- Combined L1 training (HU 800 + heavy aug): ep16/100, val Dice 0.789 (Job 8123, RUNNING)

---

## Slide 8: TMI — Architecture Comparison (ALL fold 1 COMPLETE)

| Architecture | Type | Pretrained | Val Dice | Params | Epochs |
|---|---|---|---|---|---|
| **STU-Net Large** | CNN (nnU-Net) | Random | **0.801** | ~440M | 47/50 |
| **MedNeXt-B** | CNN (ConvNeXt) | None | **0.800** | ~11M | 49/64 (early stopped) |
| **CT-FM SegResNet** | CNN (ResNet) | SSL 143K cardiac | 0.792 | ~88M | baseline |
| **SwinUNETR** | Transformer (hierarchical) | Random | 0.787 | ~62M | 44/50 |
| STU-Net Base | CNN (nnU-Net) | Sup TotalSeg 14K | 0.778 | ~60M | 30/30 |
| SwinUNETR | Transformer (hierarchical) | BTCV (30 CTs) | 0.774 | ~62M | 28/30 |
| VISTA3D | CNN (SegResNet) | Sup 11.4K, 127 cls | 0.763 | ~130M | 29/30 |
| UNETR | Transformer (flat ViT) | Random | 0.760 | ~102M | 50/50 |
| UNETR | Transformer (flat ViT) | BTCV (30 CTs) | 0.744 | ~102M | 28/30 |

**Test inference submitted for ALL 9 architectures (Job 8139, QUEUED)**

---

## Slide 9: TMI — Architecture Findings

**Top 4 architectures** (val Dice):
1. **STU-Net Large** (0.801) — brute-force capacity, 440M random-init params
2. **MedNeXt-B** (0.800) — modern efficient CNN, only 11M params, no pretraining
3. **CT-FM SegResNet** (0.792) — domain-specific SSL pretraining (143K cardiac CTs)
4. **SwinUNETR random** (0.787) — only transformer competitive with CNNs (hierarchical attention)

**Key insights**:
- **CNNs beat transformers** for fine vessels: convolutions' locality bias suits thin structures
- **Capacity or modern design compensates for no pretraining**: STU-Net-L and MedNeXt match/beat CT-FM
- **Wrong-domain pretraining hurts**: BTCV makes both SwinUNETR (-1.3pp) and UNETR (-1.6pp) worse than random init
- **Coarse-grained supervised pretraining doesn't transfer**: VISTA3D (11.4K CTs, 127 classes) = 0.763
- **Domain-specific SSL > general supervised**: CT-FM (143K cardiac) > STU-Net Base (14K whole-body) > VISTA3D (11.4K, 127 cls)
- **Hierarchical >> flat attention**: SwinUNETR (0.787) >> UNETR (0.760)

**Caveat**: These are all val Dice — test inference running (Job 8139). Val-test gap could reshuffle rankings.

---

## Slide 10: TMI — Phase 3: Cross-Architecture Validation

**Testing top 4 architectures with best preprocessing (HU 800 + heavy aug) and no-BCS controls.**

| Architecture | No BCS (control) | BCS w=0.05 (done) | BCS + HU800 + heavy aug |
|---|---|---|---|
| CT-FM SegResNet | L1 = 0.790 (done) | 0.792 (done) | L1 combo at ep16/100 (Job 8123) |
| STU-Net Large | Job 8130_0 (RUNNING) | **0.801** (done) | Job 8130_3 (RUNNING) |
| MedNeXt-B | Job 8130_1 (RUNNING) | **0.800** (done) | Job 8130_4 (RUNNING) |
| SwinUNETR Random | Job 8130_2 (RUNNING) | 0.787 (done) | Job 8130_5 (RUNNING) |

**Purpose**: Quantify BCS effect per architecture (with vs without topology loss) AND test whether preprocessing gains transfer across architectures.

---

## Slide 11: TMI — K-Fold CV Progress

L1 baselines (folds 2-4) — **resuming from HPC crash**:

| Fold | Progress | Best Val Dice | Status |
|---|---|---|---|
| 1 (EXP-028) | 100/100 | 0.790 | DONE |
| 2 | 65/100 | **0.817** (ep62) | Resuming (Job 8131_0, RUNNING) |
| 3 | 26/100 | 0.809 (ep23) | Resuming (Job 8131_1, QUEUED) |
| 4 | 70/100 | **0.818** (ep65) | Resuming (Job 8131_2, QUEUED) |

All folds above 0.80 — healthy. Once complete, all fold 1 experiments get repeated on folds 2-4.

---

## Slide 12: HPC Crash & Recovery (Mar 1)

**HPC DNS failure** knocked out entire cluster ~00:59 Mar 1. All running jobs killed.

**Impact**:
- BCS+clDice frozen: killed at ep20/30 -> resumed (Job 8122)
- L1 combo (HU800+aug): killed at ep17/100 -> resumed (Job 8123)
- L1 folds 2-4: killed at ep65/26/70 -> resumed (Job 8131)
- All fold 2/3 preprocessing/architecture experiments: killed before epoch 1 (empty checkpoints)

**Fix applied**: Added `latest_checkpoint.pth` saving (model + optimizer + scheduler + scaler + epoch + patience counter) after every validation. All future crashes are fully resumable.

**All fold 1 experiments were already complete before the crash — no data lost.**

---

## Slide 13: Methodological Fix — Inference Overlap

**Issue discovered**: Inconsistent sliding window overlap across experiments.

| Script group | Old overlap | New overlap |
|---|---|---|
| ImageCAS test inference (all MIUA models) | 0.5 | **0.75** |
| ASOCA zero-shot + threshold sweep | 0.5 | **0.75** |
| ASOCA low-shot sweep | 0.75 | 0.75 (already correct) |
| Ensemble / TTA | 0.75 | 0.75 (already correct) |
| **TMI architecture test inference** | — | **0.75** (new) |

All scripts now standardised at 0.75. TMI test inference script created (`scripts/run_tmi_test_inference.py`) — universal loader for all 9 architectures.

---

## Slide 14: Active Jobs Summary (Mar 2 evening)

| Job | What | Status | ETA |
|---|---|---|---|
| **8102** (array 0-9) | ImageCAS test overlap=0.75, 10 MIUA models | RUNNING | ~12-18h |
| **8113** | ASOCA re-run overlap=0.75 | RUNNING | ~6-12h |
| **8122** | Resume BCS+clDice ep20->30 | RUNNING | ~4h |
| **8123** | Resume L1 combo (HU800+aug) ep16->100 | RUNNING | ~3.5 days |
| **8130** (array 0-5) | No-BCS controls + best-preproc architectures | RUNNING | ~2 days each |
| **8131** (array 0-2) | Resume L1 folds 2-4 | RUNNING/QUEUED | ~1-2 days each |
| **8132** (array 0-1) | clDice test inference (frozen + full) | QUEUED | ~12h each |
| **8139** (array 0-8) | **TMI test inference, ALL 9 architectures** | QUEUED | ~12-18h each |

---

## Slide 15: Next Steps

### MIUA (deadline ~April 2)
- [ ] **Overlap fix**: Jobs 8102 (ImageCAS) + 8113 (ASOCA) running — update all metrics when done
- [ ] **clDice test inference**: Job 8132 queued — determines whether val Dice translates
- [ ] **BCS+clDice**: Job 8122 running — test inference when training completes
- [ ] Rerun CKA/probe analyses with final checkpoints
- [ ] Fill all paper tables with overlap=0.75 numbers
- [ ] Write up clDice comparison + nuance the conclusion

### TMI (target late 2026)
- [ ] **Test inference for all 9 architectures** (Job 8139) — val vs test comparison
- [ ] L1 combo (HU800+aug) resuming at ep16/100 (Job 8123) -> then L4 on top
- [ ] No-BCS control experiments (Job 8130_0-2) -> quantify BCS effect per architecture
- [ ] Best-preproc architecture experiments (Job 8130_3-5) -> cross-validate preprocessing
- [ ] Resume L1 folds 2-4 (Job 8131) -> 4-fold CV for all experiments
- [ ] nnU-Net + BCS (separate framework)
- [ ] Bootstrap significance tests across all comparisons
- [ ] CKA representation analysis across architectures

### Data access (parallel)
- Ethics/data access for ACS outcome labels — has PI initiated this?

---

## Slide 16: Questions / Discussion Points

1. **Test inference pending (Job 8139)**: Val Dice doesn't guarantee test performance — clDice historically over-promised on val (0.808) but underdelivered on test (0.787). All 9 architectures' test numbers should be ready within 48h.

2. **Top 4 architectures**: STU-Net Large (0.801), MedNeXt-B (0.800), CT-FM (0.792), SwinUNETR (0.787). Is this sufficient for the TMI comparison, or should we include more?

3. **No-BCS controls**: To claim "BCS generalises across architectures", we need per-architecture controls. Jobs running (8130). Is the 4-architecture comparison sufficient?

4. **Pretraining story**: CT-FM is the only cardiac-relevant pretraining that exists. No equivalent for SwinUNETR/STU-Net. This makes the "domain-specific pretraining matters" finding unique to CT-FM. Should we frame this as a limitation or a finding?

5. **Knowledge distillation** (future): Could transfer CT-FM's topology awareness to higher-capacity models (STU-Net-L, MedNeXt). Worth exploring for thesis, but likely out of scope for Paper 3.

6. **ACS data access**: Timeline for ethics approval and outcome labels?
