# Weekly Update — 23 Feb 2026

---

## Slide 1: Overview

**Since MICCAI submission (Feb 26):**
- Retrained all experiments after checkpoint loss (Feb 19)
- Full ablation grid: 5 adaptation strategies x 3 BCS weights (8/10 cells complete)
- CKA + linear probing analysis complete — explains why decoder-only works
- ASOCA: zero-shot, threshold sweep, and low-shot all finished
- 3 publication figures generated
- MIUA paper draft: ~95% written, figures integrated

**Next paper:** JBHI — topology -> clinical geometric accuracy

---

## Slide 2: MIUA Story

**Title:** "Topology-aware Low-shot Adaptation of CT Foundation Models for Coronary Artery Segmentation"

**Research question:** Given a pretrained CT-FM encoder + topology-aware loss, which adaptation strategy best balances overlap and connectivity?

**Key finding:** Decoder-only fine-tuning is optimal. Full fine-tuning catastrophically overfits. CKA confirms: the pretrained encoder already has topology-relevant features — the decoder just learns to use them.

**Deadline:** April 2, 2026 (MIUA, Dublin, July 20-22)

---

## Slide 3: Ablation Grid (ImageCAS, n=250)

| Adaptation | w | Dice | BCS | HD95 | Betti |
|---|---|---|---|---|---|
| L1 baseline | 0 | 0.790 | 0.690 | 10.5 | 4.70 |
| **Frozen** | 0.03 | **0.793** | 0.716 | 8.7 | 9.44 |
| **Frozen** | 0.05 | 0.792 | 0.722 | 8.8 | 9.11 |
| **Frozen** | 0.10 | 0.785 | 0.730 | 10.2 | 13.20 |
| Partial | 0.03 | **0.793** | 0.715 | 9.4 | 8.44 |
| Partial | 0.05 | 0.792 | 0.724 | **8.4** | **8.13** |
| Partial | 0.10 | 0.789 | 0.714 | 8.5 | 7.95 |
| Full | 0.10 | 0.748 | **0.739** | 33.6 | 19.85 |
| BitFit | 0.05 | pending | | | |
| Adapters | 0.05 | pending | | | |

- Higher w -> higher BCS but lower Dice. Sweet spot: w=0.03-0.05
- Frozen = Partial at same w. Partial doesn't clearly help.
- Full unfreeze: best BCS but catastrophic overfitting (6pp val-test gap)
- BitFit/Adapter inference running (Job 7805) — expect results ~Feb 24

---

## Slide 4: CKA + Probing — Why Decoder-Only Works

**CKA (representational similarity to L1 baseline):**

| Strategy | Level 0 | Level 1 | Level 2 | Level 3 | Bottleneck |
|---|---|---|---|---|---|
| Frozen | 1.00 | 0.99 | 0.99 | 0.98 | 0.88 |
| BitFit | 1.00 | 0.99 | 0.99 | 0.98 | 0.89 |
| Adapter | 1.00 | 0.99 | 0.99 | 0.97 | **0.90** |
| Partial | 1.00 | 0.99 | 0.99 | 0.97 | 0.84 |
| Full | 0.95 | 0.73 | 0.67 | 0.44 | **0.37** |

**Linear probing (bifurcation detection AUC):**
- All PEFT strategies: peak at Level 1 (AUC = 0.81) — identical to L1 baseline
- Full: peak shifts to Level 2 (AUC = 0.79) — feature reorganisation, not improvement

**Key insight:** The pretrained encoder already encodes bifurcation features at Level 1. All PEFT strategies preserve them (CKA >= 0.84). Full fine-tuning destroys them (CKA = 0.37). The decoder just learns to *route* this information for connectivity.

Adapter achieves highest bottleneck CKA (0.90) but no better segmentation — confirming the encoder is already sufficient.

---

## Slide 5: ASOCA — Cross-Domain Results (Complete)

**Zero-shot (n=40, threshold=0.5):**

| Model | Dice | BCS | HD95 | Betti |
|---|---|---|---|---|
| L1 baseline | **0.481** | 0.794 | **15.7** | **7.1** |
| Frozen w=0.03 | 0.461 | 0.821 | 17.3 | 11.3 |
| Frozen w=0.05 | 0.446 | 0.824 | 21.1 | 13.8 |
| Frozen w=0.10 | 0.430 | **0.838** | 25.9 | 21.4 |
| BitFit w=0.05 | 0.447 | 0.834 | 19.8 | 12.1 |

**Threshold sweep finding:** Raising threshold from 0.5 to 0.9 recovers +18.6pp Dice with only -1.6pp BCS loss. L1 beats all BCS-trained models at every threshold.

**Low-shot fine-tuning (from L1 baseline):**

| k | threshold | Dice | BCS | HD95 |
|---|---|---|---|---|
| Zero-shot | 0.5 | 0.481 | 0.794 | 15.7 |
| Zero-shot | 0.9 | 0.665 | 0.791 | 13.0 |
| **k=1** | 0.7 | **0.706** | **0.752** | **10.6** |
| k=5 | 0.4 | 0.695 | 0.699 | 26.8 |
| k=10 | 0.4 | 0.671 | 0.751 | 46.4 |

**Key finding:** k=1 is the sweet spot. More data paradoxically hurts — model overfits to ASOCA's sparser annotations. Starting from BCS-pretrained checkpoint (EXP-048) is *worse* than L1 (Dice 0.680, BCS 0.624).

---

## Slide 6: Figures (Generated)

Three publication-quality figures ready:

1. **Dice vs BCS scatter** — main result, shows Pareto frontier across all strategies
2. **CKA line plot** — blue PEFT band (preserved) vs. red full (destroyed), with channel counts
3. **ASOCA threshold sweep** — two-panel showing "free improvement zone" for threshold calibration

All figures designed for instant readability (no jargon, question-driven captions).

---

## Slide 7: MIUA Paper Status

**Complete:**
- Abstract, Intro, Related Work, Methods (all 5 subsections)
- Table 1 (ablation grid, 8/10 cells filled)
- Table 2 (CKA, 5 strategies with BitFit/Adapter)
- Table 3 (probe, encoder + decoder, 6 models)
- Table 4 (ASOCA zero-shot, 8 models)
- Table 5 (ASOCA low-shot, k=1/5/10)
- All prose sections, Discussion, Conclusion
- 3 figures with captions integrated

**Waiting on:**
- BitFit + Adapter test metrics (Job 7805, running ~1 day) -> fill Table 1
- Bootstrap significance tests (all comparisons)
- Final proofread and polish

**Timeline:** All results by ~Feb 25. Polish first week of March. Submit by April 2.

---

## Slide 8: Preprocessing Ablation (Apr-May)

**Question:** How do preprocessing choices affect segmentation topology?

**Factors to ablate:**
- HU windowing range (narrow cardiac vs wide vs no clipping)
- Resampling spacing (0.5mm isotropic vs native vs coarser)
- Heart ROI cropping (tight crop vs full FOV vs bounding box margin)
- Label cleaning (connected component filtering, small branch removal)
- Intensity normalization (z-score vs min-max vs percentile clipping)

**Why this matters:**
- Current pipeline uses fixed choices — never tested alternatives
- Preprocessing affects thin vessel visibility -> directly impacts topology (BCS)
- A bad HU window can clip calcified plaque or miss soft tissue boundaries
- Results inform Paper 3 pipeline design (which settings to use)

**Not a separate paper** — this is groundwork that feeds into Paper 3's methods section. But if results are interesting enough, could become a short section or supplementary.

---

## Slide 9: Paper 3 — JBHI Plan

**Title:** "Does Topology-Preserving Segmentation Improve Coronary Artery Geometric Analysis?"

**The question:** We showed BCS improves bifurcation connectivity (MICCAI) and decoder-only adaptation is optimal (MIUA). But does any of this matter clinically? Does better topology -> better stenosis measurements?

**Target:** JBHI (IEEE J. Biomedical and Health Informatics, ~7 IF)

**Key results we need:**
1. Per-segment geometry errors (L_v vs L_s), stratified by bifurcation vs non-bifurcation
2. BCS as predictor of geometric accuracy (correlation analysis)
3. Validation on ASOCA diseased cases with real stenosis
4. Clinical reader concordance (if PI arranges radiologist access)

**Datasets:** ImageCAS (n=250, synthetic stenosis perturbation) + ASOCA (n=40, real disease)

---

## Slide 10: Undergrad Project — Maryam (June-July, 8 weeks)

**Her project:** Geometry validation pipeline — does BCS predict clinical measurement accuracy?

**What she produces:**
- Segment matching code (pred centerlines -> GT centerlines)
- Per-segment geometry error tables (diameter, stenosis%, tortuosity)
- BCS vs geometry error correlation analysis
- ASOCA validation on diseased cases

**What I build before she starts (Apr-May):**
- AHA segment labeling (LAD/LCx/RCA from skeleton graph)
- Synthetic stenosis perturbation framework

**Her deliverables go directly into Paper 3 results tables.**

---

## Slide 11: Timeline

| When | What |
|---|---|
| Now - Feb 25 | BitFit/Adapter metrics, bootstrap tests |
| Feb 25 - Mar 31 | Polish MIUA paper, submit early if ready |
| **Apr 2** | **Submit MIUA** |
| Apr-May | **Preprocessing ablation** (HU range, spacing, cropping, label cleaning) |
| May-Jun | Build Paper 3 pipeline (AHA labeling, synthetic stenosis) |
| Jun-Jul | Undergrad: geometry validation + BCS correlation |
| Jul-Aug | Clinical reader study (if PI arranges) |
| Sep-Oct | Write Paper 3 |
| Nov-Dec | Submit JBHI |

---

## Discussion Points

1. **BitFit/Adapter results:** Expect to match frozen — CKA already confirms. Worth waiting or submit MIUA without them?
2. **ASOCA story:** Include in main paper or supplementary? It's interesting but makes the paper long.
3. **Ethics/data access:** Any progress on ACS outcome labels? Need to start ethics process for Paper 4.
4. **Maryam:** Confirmed for June start? Need to prepare project spec by May.
