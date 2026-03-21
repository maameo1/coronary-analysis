# PI Progress Update — Mar 7, 2026 (Tuesday meeting)

---

## Slide 1: Overview

**Title**: PhD Progress Update — Week of Mar 2–7

- Paper 1 (MICCAI): Submitted Feb 26 — under review
- Paper 2 (MIUA): Near-final — ASOCA results in, replicates running, figures done
- Paper 3 (TMI): All retrained test results in — CT-FM dominance confirmed
- **Key discovery**: BCS is a decoder-only tool — doesn't improve encoder features
- **Key discovery**: Patch-level val Dice is fundamentally misleading (experiment running)

---

## Slide 2: MIUA — Story So Far

**How should we adapt a pretrained CT foundation model for topology-aware coronary segmentation?**

**Answer**: Don't change the encoder. CT-FM's pretrained encoder already contains bifurcation-discriminative features (linear probe AUC=0.813). BCS loss improves decoder routing on ImageCAS (+3.2pp BCS with no Dice cost) but the encoder stays identical.

**Paper sections complete**:
- Ablation grid (5 strategies × 3 weights) — all test results
- CKA + linear probing analysis
- clDice comparison (BCS > clDice on test)
- ASOCA cross-domain (LR=5e-4 breakthrough)
- New Figure 4 (protocol mismatch) + Figure 5 (LR sweep)

---

## Slide 3: MIUA — Ablation Grid (ImageCAS, n=250)

**L1 baseline (no BCS): Dice=0.790, BCS=0.690**

| Adaptation | w=0.03 | w=0.05 | w=0.10 |
|---|---|---|---|
| **Frozen** (10M) | D=0.793 B=0.716 | D=0.792 B=0.722 | D=0.785 B=0.730 |
| **Partial** (27.7M) | D=0.793 B=0.715 | D=0.792 B=0.724 | D=0.789 B=0.714 |
| **Full** (87.8M) | — | — | D=0.744 (overfits) |
| **BitFit** (10M+14K) | — | D=~0.79 B=~0.72 | — |
| **Adapters** (10.2M) | — | D=~0.79 B=~0.72 | — |

**Key**: Frozen ≈ partial ≈ BitFit ≈ adapters. Full overfits catastrophically. Encoder adaptation doesn't help — BCS operates entirely in the decoder.

---

## Slide 4: BCS is a Decoder-Only Tool (NEW insight)

**Evidence**:
1. Frozen L4 encoder = L1 encoder (identical weights, CKA=1.0 by definition)
2. Linear probing: L1 encoder already has AUC=0.813 for bifurcation detection — BCS doesn't improve this
3. BCS gradient only reaches decoder (encoder frozen)
4. BCS is a no-op on ASOCA (no stub labels → gradient always zero)
5. Decoder gets overwritten during ASOCA transfer → BCS gains don't carry over

**Decisive test**: Job 8292 — partial L4 (EXP-047, CKA=0.84 at bottleneck) → ASOCA fine-tuning. If partial ≠ L1, encoder features matter. Results pending.

**Implication**: BCS is a same-domain decoder tuning tool, not a generalizable encoder improvement.

---

## Slide 5: ASOCA — LR=5e-4 Breakthrough

**Previous best** (LR=5e-6): k=1 Dice=0.706
**New best** (LR=5e-4): k=1 Dice=**0.748**, HD95=**7.55**

| k | LR | Strategy | Dice | BCS | HD95 |
|---|---|---|---|---|---|
| 1 | 5e-4 | plain | **0.748** | 0.742 | **7.55** |
| 1 | 5e-4 | dropout | 0.743 | **0.753** | 9.07 |
| 5 | 1e-4 | dropout | 0.720 | 0.712 | 21.1 |
| 10 | 1e-4 | dropout | 0.710 | 0.787 | 38.7 |

- k=1 dominates — more data overfits at high LR
- Replicates running (Job 8293, 50×k=1 + 10×k=5 + 5×k=10)
- MIUA Figures 4 (protocol mismatch) and 5 (LR sweep) already generated

---

## Slide 6: TMI — CT-FM Dominates (ALL test results in)

| Model | Stage | Val | Test Dice | Gap | BCS | HD95 |
|---|---|---|---|---|---|---|
| **CT-FM** | L1 base | 0.792 | **0.790** | **0.2pp** | 0.690 | 8.7 |
| **CT-FM** | L4 2-stg | 0.800 | **0.792** | 0.8pp | 0.722 | 8.8 |
| STU-Net Large | L1 base | 0.825 | 0.671 | 15.4pp | 0.731 | 59.3 |
| STU-Net Large | 2-stage | 0.826 | 0.689 | 13.7pp | 0.761 | 58.0 |
| SwinUNETR-R | L1 base | 0.812 | 0.700 | 11.2pp | 0.730 | 50.3 |
| SwinUNETR-R | 2-stage | 0.811 | 0.716 | 9.5pp | 0.739 | 49.5 |
| MedNeXt-B | L1 base | — | 0.650 | — | 0.711 | 57.0 |
| MedNeXt-B | 2-stage | 0.804 | 0.673 | 13.1pp | 0.728 | 53.2 |

CT-FM: 0.2pp gap, HD95=8.8mm. Everything else: 10-15pp gap, HD95 50-66mm.

---

## Slide 7: TMI — Why CT-FM Dominates

**Root cause**: Domain-specific pretraining = implicit cardiac ROI

- Random-init models segment coronaries correctly BUT also everything vessel-like in the chest
- STU-Net-L case study: TP=109K (≈CT-FM) but FP=95K (CT-FM: 40K)
- Patch-level val hides this — over-segmentation only visible at full-volume test

**Full-volume val experiment** (Job 8294, running):
| Model | Patch Val | Full-Vol Val (pending) | Test |
|---|---|---|---|
| STU-Net Large | 0.825 | ? | 0.671 |
| SwinUNETR-R | 0.812 | ? | 0.700 |
| CT-FM (control) | 0.792 | ? | 0.790 |

If full-vol val ≈ test, proves the evaluation method is the issue.

---

## Slide 8: TMI — 2-Stage BCS Effect by Architecture

| Architecture | L1 Dice | L4 Dice | Δ Dice | L1 BCS | L4 BCS | Δ BCS |
|---|---|---|---|---|---|---|
| **CT-FM** | 0.790 | **0.792** | +0.2pp | 0.690 | **0.722** | **+3.2pp** |
| **STU-Net** | 0.671 | **0.689** | +1.8pp | 0.731 | **0.761** | **+3.0pp** |
| **SwinUNETR** | 0.700 | **0.716** | +1.6pp | 0.730 | 0.739 | +0.9pp |
| **MedNeXt** | 0.650 | 0.673 | +2.3pp | 0.711 | 0.728 | +1.7pp |

- BCS helps all architectures (Dice +1.6–2.3pp, BCS +0.9–3.2pp)
- But doesn't fix the fundamental over-segmentation problem (HD95 still 50-66mm)
- CT-FM is the only one where BCS produces clinically acceptable output

---

## Slide 9: TMI — Preprocessing: HU800 is Safe, Augmentation Hurts

| CT-FM Variant | Test Dice | BCS | HD95 |
|---|---|---|---|
| **Base L1** (default) | **0.790** | 0.690 | **8.7** |
| HU800-only (frozen) | **0.789** | 0.719 | 10.5 |
| Aug-heavy-only (frozen) | 0.775 | 0.737 | **18.6** |
| HU800+aug combo | 0.728 | 0.738 | **47.4** |

- HU800 alone: −0.1pp Dice — essentially neutral
- Heavy augmentation alone: −1.5pp Dice, HD95 doubles
- Combined: catastrophic (−6.2pp, HD95 5× baseline)
- Full-training retrain running (Job 8287, ep31/100) to confirm with `freeze: full`

---

## Slide 10: Active Jobs & Status

| Job | Description | Status |
|---|---|---|
| 8287 | Preprocessing retrain (freeze:full) + L4 + test | ep31/100, both running |
| 8292 | L4→ASOCA transfer (partial EXP-047) | Just started |
| 8293 | ASOCA replicates at LR=5e-4 (65 runs) | Just submitted |
| 8294 | Full-volume val inference (3 models) | Just submitted |

---

## Slide 11: MIUA — What's Left

| Task | Status |
|---|---|
| Ablation grid (ImageCAS) | DONE |
| CKA + linear probing | DONE |
| clDice comparison | DONE |
| ASOCA LR sweep | DONE |
| ASOCA replicates (variance) | RUNNING (Job 8293, ~22h) |
| L4→ASOCA transfer test | RUNNING (Job 8292, ~12h) |
| Protocol mismatch figure | DONE |
| LR sweep figure | DONE — needs update with k=5/k=10 |
| Update paper tables with final ASOCA numbers | TODO |
| Write/revise ASOCA results section | TODO |

**Deadline**: ~April 2. On track.

---

## Slide 12: TMI — What's Left

| Task | Status |
|---|---|
| All retrained L1 + 2-stage test inference | DONE (10 models) |
| CT-FM preprocessing isolated effects | Frozen: DONE. Full: RUNNING (8287) |
| Full-volume val experiment | RUNNING (Job 8294) |
| K-fold CV (L4 for folds 2-4) | NOT STARTED |
| nnU-Net + BCS | NOT STARTED |
| Bootstrap significance tests | NOT STARTED |
| CKA across architectures | NOT STARTED |
| Paper writing | NOT STARTED |

**Target**: September 2026 submission.

---

## Slide 13: Discussion Points

1. **BCS is decoder-only**: Does this change the MICCAI/MIUA framing? We claimed BCS improves "topology-aware segmentation" — it does, but only via decoder routing, not encoder representation learning. Is this a limitation or a finding?

2. **Patch-level val is misleading**: Should we propose full-volume validation as a methodological recommendation in TMI? This affects the entire community.

3. **CT-FM dominance**: The story is almost too clean — CT-FM wins everything. Should TMI focus on "why pretraining matters" rather than "BCS generalizes"? Because BCS helps all architectures modestly, but only CT-FM produces clinically useful output.

4. **ASOCA k=1 LR=5e-4**: Dice=0.748 from just 2 training scans is remarkable. Worth highlighting as a practical contribution?

5. **ACS data access**: Timeline for ethics approval and outcome labels? (Paper 4)
