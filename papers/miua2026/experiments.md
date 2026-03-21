# MIUA 2026 — Experiment Tracker

**Title**: Topology-aware Low-shot Adaptation of CT Foundation Models for Coronary Artery Segmentation
**Venue**: MIUA 2026, UCD Dublin, July 20–22. Springer LNCS, 8–15 pages.
**Deadline**: ~April 2, 2026
**Core question**: How does encoder adaptation strategy interact with BCS loss weight? Does it generalise under annotation protocol mismatch?

---

## Status Summary (updated Mar 10)

**MIUA paper nearly complete.** All numbers verified against source JSON files. All tables filled. Bootstrap significance added. Figures generated.

**Completed since Mar 2:**
- All ImageCAS test inference (overlap=0.75) — DONE
- All ASOCA zero-shot + threshold sweep (overlap=0.75) — DONE
- clDice frozen + full: trained, tested, in paper — DONE
- Full+BCS corrected (LR=1e-5, matched settings) — DONE (Dice=0.797, BCS=0.735)
- Fully Soft BCS: implemented, trained, tested — DONE (Dice=0.782, BCS=0.729)
- CKA analysis: all 11 models — DONE
- Linear probing (encoder + decoder): all models incl. matched Full — DONE
- Bootstrap significance (10,000 resamples): all experiments vs L1 — DONE
- ASOCA LR sweep: all k, 6 LRs, ± dropout — DONE
- ASOCA replicates: 50/10/5 reps trained (k=1/5/10) — k=1 eval 4/5 done
- Architecture visual comparison figures — DONE


**Active jobs (Mar 10):**
- **Job 8338**: ASOCA replicate eval v2 — running k=10 replicates (>28h, nearly done)
**Remaining before submission (deadline ~April 2):**
- Update replicate error bars when Job 8338 finishes
- Final proofread
- Send to PI for review

---

## Ablation Grid: Adaptation Strategy × BCS Weight

All experiments fine-tune from CT-FM L1 (vesselness) checkpoint on ImageCAS-638.
Test set: N=250. Inference: overlap=0.75, mode=constant.

All inference at overlap=0.75. Test set: N=250.

| Adaptation | λ_s | LR | Dice | clDice | BCS | HD95 | Betti | Exp |
|---|---|---|---|---|---|---|---|---|
| *L1 baseline* | 0 | — | 0.790 | 0.859 | 0.688 | 6.7 | 5.0 | EXP-028 |
| Frozen | 0.03 | 1e-5 | **0.794** | 0.864 | 0.705 | 7.2 | 6.8 | EXP-044 |
| Frozen | 0.05 | 1e-5 | 0.792 | 0.856 | 0.722 | 8.8 | 9.1 | EXP-048 |
| Frozen | 0.10 | 1e-5 | 0.785 | 0.844 | 0.730 | 10.2 | 13.2 | EXP-046 |
| Partial | 0.03 | 5e-5 | 0.793 | 0.858 | 0.715 | 9.4 | 8.4 | EXP-049 |
| Partial | 0.05 | 5e-5 | 0.793 | 0.862 | 0.719 | 7.8 | 8.2 | EXP-047 |
| Partial | 0.10 | 5e-5 | 0.789 | 0.858 | 0.714 | 8.5 | 8.0 | EXP-050 |
| BitFit | 0.05 | 1e-5 | 0.793 | 0.859 | 0.721 | 8.1 | 8.7 | EXP-052 |
| Adapters | 0.05 | 1e-5 | 0.790 | 0.852 | 0.714 | 9.3 | 10.6 | EXP-053 |
| Full (BCS) | 0.10 | 1e-5 | 0.797 | 0.866 | **0.735** | **6.7** | 5.7 | Full+BCS matched |
| Frozen (clDice) | 0.05 | 1e-5 | 0.794 | 0.861 | 0.706 | 9.1 | 9.1 | clDice frozen |
| Full (clDice) | 0.10 | 1e-5 | **0.801** | **0.872** | 0.717 | 6.8 | **5.0** | clDice full |
| ~~Full (confounded)~~ | ~~0.10~~ | ~~1e-4~~ | ~~0.748~~ | ~~0.778~~ | ~~0.739~~ | ~~33.6~~ | ~~19.9~~ | ~~EXP-045~~ |

**Key findings:**
- Full+clDice best Dice (0.801); Full+BCS best BCS (0.735). Adaptation strategy > loss choice.
- EXP-045 was confounded (10× LR) — not BCS overfitting, just hyperparameter error.
- All BCS improvements significant vs L1 (bootstrap p<0.001). clDice Dice p=0.022.
- Fully Soft BCS (no precomputation): Dice=0.782, BCS=0.729 — validates on-the-fly approach.

---

## Experiment Details

### Baseline: CT-FM L1 (EXP-028)
- **Test (n=250, overlap=0.75)**: Dice=0.790, clDice=0.859, BCS=0.688, HD95=6.7, Betti=5.0
- **Checkpoint**: `experiments/EXP-028_L1_vesselness_ctfm_638cases_20260109_043910/`

### EXP-044 — Frozen, λ=0.03
- **Test**: Dice=0.794, clDice=0.864, BCS=0.705, HD95=7.2, Betti=6.8
- **ASOCA zero-shot**: Dice=0.461, BCS=0.821, HD95=17.3, Betti=11.3

### EXP-048 — Frozen, λ=0.05
- **Test**: Dice=0.792, clDice=0.856, BCS=0.722, HD95=8.8, Betti=9.1
- **ASOCA zero-shot**: Dice=0.446, BCS=0.824, HD95=21.1, Betti=13.8

### EXP-046 — Frozen, λ=0.10
- **Test**: Dice=0.785, clDice=0.844, BCS=0.730, HD95=10.2, Betti=13.2
- **ASOCA zero-shot**: Dice=0.430, BCS=0.838, HD95=25.9, Betti=21.4

### EXP-049 — Partial, λ=0.03
- **Test**: Dice=0.793, clDice=0.858, BCS=0.715, HD95=9.4, Betti=8.4
- **ASOCA zero-shot**: Dice=0.453, BCS=0.815, HD95=22.6, Betti=14.5

### EXP-047 — Partial, λ=0.05
- **Test**: Dice=0.793, clDice=0.862, BCS=0.719, HD95=7.8, Betti=8.2
- **ASOCA zero-shot**: Dice=0.431, BCS=0.829, HD95=21.9, Betti=14.8

### EXP-050 — Partial, λ=0.10
- **Test**: Dice=0.789, clDice=0.858, BCS=0.714, HD95=8.5, Betti=8.0
- **ASOCA zero-shot**: Dice=0.446, BCS=0.830, HD95=20.4, Betti=14.4

### EXP-045 — Full, λ=0.10 (CONFOUNDED — LR=1e-4)
- **Test**: Dice=0.748, clDice=0.778, BCS=0.739, HD95=33.6, Betti=19.9
- **Notes**: 10× LR caused catastrophic overfitting. Replaced by Full+BCS matched (LR=1e-5).

### Full+BCS matched — Full, λ=0.10, LR=1e-5
- **Test**: Dice=0.797, clDice=0.866, BCS=0.735, HD95=6.7, Betti=5.7
- **Notes**: Corrected version of EXP-045. Best BCS in ablation.

### EXP-052 — BitFit, λ=0.05
- **Test**: Dice=0.793, clDice=0.859, BCS=0.721, HD95=8.1, Betti=8.7
- **ASOCA zero-shot**: Dice=0.454, BCS=0.834, HD95=20.1, Betti=9.0

### EXP-053 — Adapters, λ=0.05
- **Test**: Dice=0.790, clDice=0.852, BCS=0.714, HD95=9.3, Betti=10.6
- **ASOCA zero-shot**: Dice=0.454, BCS=0.835, HD95=20.0, Betti=12.4

### clDice Frozen — Frozen, clDice w=0.05
- **Test**: Dice=0.794, clDice=0.861, BCS=0.706, HD95=9.1, Betti=9.1

### clDice Full — Full, clDice w=0.10, LR=1e-5
- **Test**: Dice=0.801, clDice=0.872, BCS=0.717, HD95=6.8, Betti=5.0
- **Notes**: Best Dice in entire ablation.

### Fully Soft BCS — Frozen, BCS w=0.05 (no precomputed stubs)
- **Test**: Dice=0.782, BCS=0.729, HD95=10.2, Betti=13.1
- **Notes**: BCS computed on-the-fly from raw GT. Eliminates 775GB precomputation.

---

## Representational Analyses — ALL COMPLETE

### CKA (Centered Kernel Alignment)
- **Status**: DONE — all 11 models (9 PEFT + 2 matched full + confounded)
- **Location**: `miua2026/results/cka/cka_results.json`
- **Three-tier structure**:
  - PEFT (frozen/partial/BitFit/adapter): CKA ≥ 0.83 at bottleneck
  - Full FT at matched LR (1e-5): CKA ≈ 0.75 at bottleneck
  - Confounded Full FT (1e-4): CKA = 0.37 at bottleneck
- **In paper**: Table 3 + Figure 2

### Linear Probe (Encoder + Decoder)
- **Status**: DONE — all models incl. Full+BCS/clDice matched
- **Location**: `miua2026/results/probe_results/`
- **Key finding**: All non-confounded encoders peak at L1 (AUC≈0.81). Confounded (1e-4) shifts peak to L2.
  - Old claim ("full FT shifts peak to L2") was a LR artifact.
  - All non-confounded decoders peak at D2 (AUC≈0.75-0.76).
- **In paper**: Table 4

### Bootstrap Significance
- **Status**: DONE — 10,000 resamples, all experiments vs L1
- **Location**: `miua2026/results/bootstrap_all_vs_l1.json`
- **Key**: All BCS improvements p<0.001. Full+clDice Dice p=0.022. Frozen λ=0.03 Dice p<0.001.

---

## Figures — ALL GENERATED

| Fig | Description | Script | Status |
|---|---|---|---|
| 1 | Architecture diagram | Manual (TikZ) | DONE |
| 2 | CKA heatmap (11 models × 5 levels) | `scripts/plot_cka_heatmap.py` | DONE |
| 3 | Probe AUC line plot (enc+dec) | `scripts/plot_probes.py` | DONE |
| 4 | ASOCA protocol mismatch (FP% vis) | `scripts/visualize_asoca_protocol_mismatch.py` | DONE |
| 5 | ASOCA LR sweep (Dice + BCS panels) | `scripts/plot_asoca_lr_sweep.py` | DONE |

### Statistical Tests — DONE
- Paired bootstrap (10,000 resamples): all experiments vs L1
- **Location**: `miua2026/results/bootstrap_all_vs_l1.json`

---

## ASOCA Cross-Domain Transfer

### ASOCA Dataset
- 40 cases: 20 Normal + 20 Diseased (NRRD format)
- Split: k+k train, 2+2 val, rest eval (varies by k)
- Annotations: major vessels only (no side branches)
- Protocol mismatch: ImageCAS = full tree, ASOCA = major vessels

### ASOCA Zero-Shot Results (n=40, overlap=0.75) — FINAL

| Model | Dice | BCS | HD95 | Betti |
|---|---|---|---|---|
| L1 baseline (no BCS) | **0.481** | 0.794 | **15.7** | **7.1** |
| Frozen w=0.03 (EXP-044) | 0.461 | 0.821 | 17.3 | 11.3 |
| Frozen w=0.05 (EXP-048) | 0.446 | 0.824 | 21.1 | 13.8 |
| Frozen w=0.10 (EXP-046) | 0.430 | **0.838** | 25.9 | 21.4 |
| BitFit w=0.05 (EXP-052) | 0.454 | 0.834 | 20.1 | 9.0 |
| Partial w=0.03 (EXP-049) | 0.453 | 0.815 | 22.6 | 14.5 |
| Partial w=0.05 (EXP-047) | 0.431 | 0.829 | 21.9 | 14.8 |
| Partial w=0.10 (EXP-050) | 0.446 | 0.830 | 20.4 | 14.4 |
| Adapters w=0.05 (EXP-053) | 0.454 | 0.835 | 20.0 | 12.4 |
| Full w=0.10 (EXP-045) | 0.423 | 0.831 | 43.0 | 21.6 |

**Key finding**: BCS training helps BCS on ASOCA (+3-4pp) but hurts Dice (-2 to -6pp). Protocol mismatch: model preserves side branches ASOCA doesn't annotate.

### Threshold Sweep (zero-shot L1) — FINAL

| Thresh | Dice | BCS | HD95 |
|---|---|---|---|
| 0.5 | 0.481 | 0.794 | 15.7 |
| 0.9 | 0.671 | 0.801 | 13.0 |

Threshold 0.9 recovers +19pp Dice with only +0.7pp BCS change.

### Low-Shot Fine-Tuning (from L1, encoder frozen) — FINAL

| k | LR | Reg | n_eval | Dice | BCS | HD95 |
|---|---|---|---|---|---|---|
| 1 | 5e-6 | — | 34 | 0.658 | 0.761 | 11.0 |
| 1 | 5e-4 | — | 34 | **0.748** | 0.742 | **7.6** |
| 1 | 5e-4 | dp | 34 | 0.743 | **0.753** | 9.1 |
| 1‡ | 5e-4 | — | 34 | 0.706±.032 | 0.746±.027 | 35.0±11.3 |
| 5 | 5e-6 | — | 26 | 0.687 | 0.672 | 24.1 |
| 5 | 1e-4 | dp | 26 | **0.720** | **0.712** | **21.1** |
| 10 | 5e-6 | — | 16 | 0.680 | 0.733 | 44.0 |
| 10 | 1e-4 | dp | 16 | **0.710** | **0.787** | **38.7** |

‡ Mean±SD over 4 replicates (5th completing in Job 8338). k=5/k=10 replicates also pending.

**Key findings:**
- LR is the dominant lever: 5e-6 → 5e-4 at k=1 gives +9pp Dice, −3.4mm HD95
- k=1 best balance across all metrics; more data doesn't help (protocol mismatch)
- clDice neutral during ASOCA fine-tuning (Dice~0.744–0.748, BCS~0.734–0.744)
- BCS-trained decoder init hurts transfer (Dice~0.638 vs 0.748 from L1)

---

## Additional Experiments

### clDice Comparison — COMPLETE
- Frozen clDice (w=0.05): Dice=0.794, BCS=0.706 — matches frozen BCS
- Full clDice (w=0.10, LR=1e-5): Dice=0.801, BCS=0.717 — best Dice overall
- Narrative: "adaptation strategy > loss choice" — BCS and clDice equivalent under matched settings

### Fully Soft BCS — COMPLETE
- Frozen, w=0.05, no precomputed stubs: Dice=0.782, BCS=0.729
- Eliminates 775GB precomputation; BCS now works on any dataset
- PhD contribution (not in MIUA paper main results, noted in discussion)

### Full+BCS Corrected (LR Confound) — COMPLETE
- Old EXP-045 used LR=1e-4 (10× too high) → catastrophic overfitting
- Corrected run at LR=1e-5: Dice=0.797, BCS=0.735 — best BCS in ablation
- The 10× LR was the entire problem, not the loss function

---

## Action Items — updated Mar 10

### Completed
- [x] All ablation training (10 BCS + 2 clDice + Full+BCS matched + Fully Soft BCS)
- [x] All ImageCAS test inference (overlap=0.75, n=250)
- [x] All ASOCA zero-shot + threshold sweep (overlap=0.75, n=40)
- [x] ASOCA low-shot LR sweep (all k, 6 LRs, ± dropout)
- [x] CKA analysis (all 11 models)
- [x] Linear probing (encoder + decoder, all models incl matched Full)
- [x] Bootstrap significance (10,000 resamples, all vs L1)
- [x] All figures generated (CKA heatmap, probes, LR sweep, protocol mismatch)
- [x] Paper text: all tables filled, numbers verified, significance added
- [x] Narrative reframed: "adaptation strategy > loss choice"
- [x] clDice comparison integrated into results + discussion
- [x] Abstract + contributions updated

### In progress
- [ ] **ASOCA replicate eval** (Job 8338) — k=10 replicates running (~28h in)

### Remaining before submission (deadline ~April 2)
- [ ] Update replicate mean±SD when Job 8338 finishes
- [ ] Final proofread
- [ ] Send to PI for review
