# MIUA 2026 Paper Plan

**Title**: Topology-aware Low-shot Adaptation of CT Foundation Models for Coronary Artery Segmentation
**Venue**: MIUA 2026, UCD Dublin, July 20–22. Springer LNCS, 8–15 pages.
**Deadline**: ~April 2, 2026
**Status**: Draft written, figures scripted, waiting on retraining + ASOCA experiments

---

## The Story

**One sentence**: Freezing the encoder and training only the decoder is the best way to add topology-aware supervision to a pretrained CT foundation model — and this strategy also enables robust low-shot adaptation under annotation protocol mismatch.

**Two-part narrative**:

1. **Part A (ImageCAS ablation)**: *How* should you adapt a foundation model for topology? We cross 5 adaptation strategies × 3 BCS loss weights → decoder-only wins. CKA + linear probing explain *why*: the pretrained encoder already has bifurcation-relevant features, full fine-tuning destroys them.

2. **Part B (ASOCA transfer)**: *Does it generalise?* Zero-shot: SoftBCS models over-segment on ASOCA (+3-4pp BCS but -3-5pp Dice) because they find branches ASOCA doesn't label. Low-shot: lighter fine-tuning (last 2 decoder layers) teaches the model to "unlearn" extra branches. Threshold tuning as a zero-cost alternative.

**PI insight**: "If we're oversegmenting, it's a case of ignoring vessels rather than learning something new." This frames Part B — the task is suppression, not acquisition.

---

## Sections & Status

| # | Section | Status | Notes |
|---|---------|--------|-------|
| 1 | Abstract | DRAFT | Covers both parts. ~200 words. |
| 2 | Introduction | DRAFT | Two challenges: adaptation strategy + protocol mismatch. 3 contributions. |
| 3 | Related Work | DRAFT | FM adaptation + topology-aware segmentation. Gap identified. |
| 4 | Methods | DRAFT | 6 subsections (BCS, base model, 5 strategies, CKA/probing, ASOCA protocol, metrics) |
| 5 | Results | PARTIAL | Tables 1–4 have real data. Tables 5–6 (ASOCA) are placeholders. |
| 6 | Discussion | DRAFT | 4 paragraphs + limitations. |
| 7 | Conclusion | DRAFT | Tight summary. |

---

## Experiments & Data

### Part A — ImageCAS Ablation (8/10 complete)

| ID | Experiment | Dice | BCS | HD95 | Betti | Status |
|----|-----------|------|-----|------|-------|--------|
| L1 | No BCS, fully trained | 0.790 | 0.690 | 17.5 | 4.70 | DONE |
| EXP-044 | Frozen, λ=0.03 | 0.793 | 0.716 | 8.7 | 9.44 | RETRAINING |
| EXP-048 | Frozen, λ=0.05 | 0.792 | 0.722 | 8.8 | 9.11 | RETRAINING ep20 |
| EXP-046 | Frozen, λ=0.10 | 0.785 | 0.730 | 10.2 | 13.20 | RETRAINING ep19 |
| EXP-049 | Partial, λ=0.03 | 0.793 | 0.715 | 9.4 | 8.44 | RETRAINING ep20 |
| EXP-047 | Partial, λ=0.05 | 0.792 | 0.724 | 8.4 | 8.13 | RETRAINING ep19 |
| EXP-050 | Partial, λ=0.10 | 0.789 | 0.714 | 8.5 | 7.95 | RETRAINING ep6 |
| EXP-045 | Full, λ=0.10 | 0.748 | 0.739 | 33.6 | 19.85 | RETRAINING ep6 |
| EXP-052 | BitFit, λ=0.05 | — | — | — | — | RETRAINING ep19 |
| EXP-053 | Adapter, λ=0.05 | — | — | — | — | RETRAINING ep6 |

### Part B — ASOCA Cross-Domain

| Experiment | Status | Script | HPC |
|---|---|---|---|
| Zero-shot all models (40 cases) | OLD DATA EXISTS, re-running with new ckpts | `run_asoca_zeroshot_softbcs.py` | `asoca_zeroshot_v2.sh` |
| Threshold sweep (0.3–0.9) | OLD DATA LOST, re-running | `run_asoca_threshold_sweep.py` | `asoca_threshold_sweep_v2.sh` |
| Low-shot sweep (3×4=12 runs) | PLANNED — PI's lighter approach | `run_asoca_lowshot_sweep.py` | `asoca_lowshot_sweep_v2.sh` |

**ASOCA Zero-Shot (old data, survived in JSON):**

| Model | Dice | BCS | Δ Dice vs L1 | Δ BCS vs L1 |
|---|---|---|---|---|
| L1 baseline | 0.481 | 0.794 | — | — |
| Frozen w=0.03 | 0.450 | 0.827 | -0.031 | +0.034 |
| Frozen w=0.05 | 0.446 | 0.824 | -0.035 | +0.030 |
| Frozen w=0.10 | 0.430 | 0.838 | -0.051 | +0.044 |
| Partial w=0.03 | 0.453 | 0.815 | -0.028 | +0.022 |
| Partial w=0.05 | 0.426 | 0.821 | -0.055 | +0.027 |
| Partial w=0.10 | 0.446 | 0.830 | -0.035 | +0.037 |
| Full w=0.10 | 0.423 | 0.831 | -0.058 | +0.037 |

---

## Tables (6 total)

| Table | Content | Status |
|-------|---------|--------|
| 1 | Adaptation strategies (params, LR) | DONE |
| 2 | Ablation grid: 5 strategies × 3 weights, 6 metrics | 8/10 cells filled |
| 3 | CKA similarity: 4 models × 5 encoder levels | DONE (pre-recovery) |
| 4 | Linear probe AUC: encoder (5×5) + decoder (5×4) | DONE (pre-recovery) |
| 5 | ASOCA zero-shot: L1 + best models | OLD DATA EXISTS, re-running |
| 6 | ASOCA low-shot: k × model, 6 metrics | PENDING low-shot sweep |

## Figures (6 planned)

| Fig | Description | Script | Status |
|---|---|---|---|
| 1 | Pareto scatter (Dice vs BCS) | `miua2026/scripts/plot_pareto.py` | SCRIPT READY |
| 2 | CKA heatmap | `miua2026/scripts/plot_cka_heatmap.py` | SCRIPT READY |
| 3 | Probe AUC line plot | `miua2026/scripts/plot_probes.py` | SCRIPT READY |
| 4 | Convergence curves | TODO | BLOCKED on training logs |
| 5 | Qualitative examples | `scripts/visualize_case.py` | BLOCKED on predictions |
| 6 | ASOCA learning curve | TODO | BLOCKED on low-shot sweep |

---

## Timeline

### Week of Feb 21–27
- [x] Parallel retraining all 8 experiments
- [x] Figure scripts written (Pareto, CKA, probes)
- [x] ASOCA scripts updated with dynamic checkpoint resolver
- [x] ASOCA HPC scripts created (zero-shot, threshold sweep, low-shot)
- [ ] Training completes → inference runs (job 7805)
- [ ] Submit ASOCA zero-shot + threshold sweep
- [ ] Fill Table 2 with final test metrics
- [ ] Run bootstrap significance tests

### Week of Feb 28 – Mar 6
- [ ] ASOCA low-shot sweep runs
- [ ] Rerun CKA/probe with new checkpoints (if desired)
- [ ] Generate all figures
- [ ] Fill ASOCA tables
- [ ] Write ASOCA results narrative

### Week of Mar 7–13
- [ ] Complete all TODO cells in paper
- [ ] Update abstract + discussion with final numbers
- [ ] Polish introduction + related work
- [ ] Page count check

### Week of Mar 14–20
- [ ] Final proofread
- [ ] Send to PI for review

### Week of Mar 21–April 2
- [ ] PI feedback + revisions
- [ ] Submit

---

## Key Scripts

| Script | Purpose |
|---|---|
| `scripts/asoca_checkpoint_resolver.py` | Find latest checkpoint for any experiment |
| `scripts/run_asoca_zeroshot_softbcs.py` | Zero-shot eval all models on ASOCA |
| `scripts/run_asoca_threshold_sweep.py` | Binarization threshold sweep on ASOCA |
| `scripts/run_asoca_lowshot_sweep.py` | Low-shot fine-tuning sweep (with `--freeze_decoder_depth`) |
| `miua2026/scripts/plot_pareto.py` | Fig 1: Dice vs BCS scatter |
| `miua2026/scripts/plot_cka_heatmap.py` | Fig 2: CKA heatmap |
| `miua2026/scripts/plot_probes.py` | Fig 3: Encoder + decoder probe AUC |
| `scripts/bootstrap_l4_vs_l1.py` | Paired bootstrap significance tests |

---

## Risks

| Risk | Mitigation |
|------|------------|
| BitFit/Adapters match frozen exactly | Still informative — confirms encoder modification is unnecessary |
| ASOCA low-shot: BCS loss saturates again | PI's lighter approach (freeze early decoder) + lower BCS weight |
| ASOCA low-shot: topo-aware doesn't beat Dice-only | Report honestly; discuss why |
| Page count exceeds 15 | Move probe table or CKA to supplementary |
| New checkpoints give different numbers | Expected to be very close; update tables |
| EXP-045 (100 epochs) doesn't finish in time | Have pre-recovery metrics |
