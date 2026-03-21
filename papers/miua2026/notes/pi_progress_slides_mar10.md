# PI Progress Update — Mar 10, 2026 (Tuesday meeting)

---

## Slide 1: Since March 2

**Week of Mar 2–9: Major progress on MIUA paper + new experiments**

| Item | Status Mar 2 | Status now |
|---|---|---|
| Full+BCS "overfitting" | Assumed loss-specific problem | **LR confound found + fixed**: matched settings → Dice=0.797, BCS=0.735 |
| Fully Soft BCS | Concept only | **Implemented, trained, tested**: Dice=0.782, BCS=0.729 (no precomputation!) |
| Fully Soft BCS → ASOCA | N/A | **Tested**: BCS hurts ASOCA (Dice 0.679) — confirms protocol mismatch |
| clDice comparison | Not started | **Done**: frozen + full, ImageCAS + ASOCA. BCS ≈ clDice under matched conditions |
| ASOCA LR sweep | k=1 only, default LR | **Complete**: all k, 6 LRs each, ± dropout. k=1 at 5e-4 = sweet spot |
| ASOCA replicates | Not started | **50/10/5 reps trained**; k=1 eval 4/5 done (Job 8338 completing rest) |
| Architecture visuals | Not done | **Done**: architecture comparison + BCS vs no-BCS figures |
| Paper narrative | "BCS overfits under full FT" | **Reframed**: "Adaptation strategy > loss choice" — text tightened |
| CKA + linear probing | Done | In paper (§4.2, §4.3) |

---

## Slide 2: Key Result — LR Confound Resolved

The old claim ("BCS overfits under full fine-tuning") was a **hyperparameter confound**:

| Run | Loss | LR | Init | Test Dice | BCS | HD95 |
|---|---|---|---|---|---|---|
| EXP-045 (old) | BCS w=0.10 | **1e-4** | CT-FM pretrained | 0.748 | 0.689 | 33.6 |
| clDice full | clDice w=0.10 | 1e-5 | L1 checkpoint | **0.801** | 0.717 | 6.8 |
| **Corrected** | **BCS w=0.10** | **1e-5** | **L1 checkpoint** | **0.797** | **0.735** | **6.7** |

At matched settings (LR=1e-5, 30 epochs, L1 init):
- Full+BCS trails Full+clDice slightly on Dice (0.797 vs 0.801)
- Full+BCS **beats** clDice on BCS (+1.8pp: 0.735 vs 0.717) and HD95 (6.7 vs 6.8)
- The 10× LR was the entire problem, not the loss function
- Bootstrap: BCS improvements significant (p<0.001); clDice Dice improvement p=0.022

*(Final numbers, n=250)*

---

## Slide 3: Key Result — Fully Soft BCS

**What**: BCS loss computed on-the-fly from raw GT masks. No precomputed stubs needed.

| Metric | Precomputed BCS (EXP-048) | Fully Soft BCS | Gap |
|---|---|---|---|
| Dice | 0.792 | 0.782 | −1.0pp |
| BCS | 0.722 | **0.729** | **+0.7pp** |
| HD95 | 8.8 | 10.2 | +1.4mm |
| Betti | 9.1 | 13.1 | +4.0 |

*(Final numbers, n=250)*

- BCS **higher** than precomputed version — on-the-fly stubs find bifurcations in every patch
- Dice gap is small (1pp) and HD95 slightly worse but still single-digit
- **Eliminates 775GB precomputation requirement** — BCS now works on ANY dataset
- PhD contribution: makes BCS loss practical/dataset-agnostic like clDice

---

## Slide 4: Updated Ablation Grid (ImageCAS, n=250)

| Adaptation | Loss | λ_s | Dice | BCS | HD95 | Betti |
|---|---|---|---|---|---|---|
| *L1 baseline* | — | 0 | 0.792 | 0.688 | **6.7** | **5.0** |
| Frozen | BCS | 0.03 | **0.794** | 0.705 | 7.2 | 6.8 |
| Frozen | BCS | 0.05 | 0.792 | 0.722 | 8.8 | 9.1 |
| Frozen | BCS | 0.10 | 0.785 | **0.730** | 10.2 | 13.2 |
| Partial | BCS | 0.05 | 0.793 | 0.719 | 7.8 | 8.2 |
| **Full** | **BCS** | **0.10** | **0.797** | **0.735** | **6.7** | **5.7** |
| Frozen | clDice | 0.05 | 0.794 | 0.706 | 9.1 | 9.1 |
| **Full** | **clDice** | **0.10** | **0.801** | 0.717 | 6.8 | **5.0** |

**Takeaway**: Full+clDice has best Dice (0.801); Full+BCS has best BCS (0.735). Both dramatically better than confounded EXP-045. Adaptation strategy > loss choice.

---

## Slide 5: ASOCA Low-Shot — Tightened Narrative

**Paper §4.5 rewritten to three clean paragraphs:**

1. **LR is the lever**: Default LR underfits; 5e-4 at k=1 → Dice 0.748, HD95 7.6mm, BCS 0.742
2. **Dice–BCS dissociation**: k=1 Dice rises with LR while BCS stays flat; k=10 BCS climbs to 0.819 while Dice drops. k=1 is the practical sweet spot.
3. **Frozen encoder drives it**: clDice is neutral on ASOCA; BCS-trained decoder init hurts transfer. It's the strategy, not the loss.

### LR sweep figure: now Dice + BCS panels (was Dice + HD95)

BCS panel tells the topology story better than HD95 — directly aligned with the paper's central metric.

---

## Slide 6: ASOCA Replicate Variance

### Val Dice (from training, n=50/10/5)

| k | Val Dice (mean±std) | Range |
|---|---|---|
| 1 | 0.759 ± 0.051 | 0.633 – 0.820 |
| 5 | 0.807 ± 0.023 | 0.764 – 0.839 |
| 10 | 0.817 ± 0.018 | 0.786 – 0.829 |

### Test-set eval (k=1, 4/5 replicates done, t=0.5)

| Rep | Dice | BCS | HD95 |
|---|---|---|---|
| r0 | 0.712 | 0.706 | 35.0 |
| r1 | 0.738 | 0.764 | 22.5 |
| r2 | 0.661 | 0.757 | 49.9 |
| r3 | 0.711 | 0.758 | 32.6 |
| **Mean±SD** | **0.706±0.032** | **0.746±0.026** | **35.0±11.5** |

- Best single run (cherry-picked): Dice=0.748, HD95=7.6mm
- Replicate mean much lower — k=1 highly sensitive to case selection
- Job 8338 completing r4 + all k=5/k=10 replicates (2-day limit)

---

## Slide 7: Architecture Visual Comparison

Two figures generated:

1. **Architecture comparison** (`architecture_visual_inspection.pdf`)
   - CT-FM vs STU-Net vs SwinUNETR vs MedNeXt on 4 test cases
   - Coronal MIP, green=TP/red=FN/blue=FP
   - CT-FM is clearly dominant — tight predictions, minimal FP

2. **BCS vs no-BCS per architecture** (`bcs_vs_nobcs_visual.pdf`)
   - BCS effect per architecture:

| Architecture | ΔBCS | ΔDice |
|---|---|---|
| CT-FM | +3.4pp | +0.2pp |
| STU-Net | +3.0pp | +1.8pp |
| SwinUNETR | +0.9pp | +1.6pp |
| MedNeXt | +1.7pp | +2.3pp |

BCS improves connectivity across all architectures, but only CT-FM is clinically usable (HD95 < 10mm).

---

## Slide 8: Val-Test Gap Root Cause

**PI question from last week**: Is there a threshold bug?

**Answer: No bug.** Both use softmax + argmax.

**Real cause**: Val = single random 96³ patch (biased to foreground). Test = full volume sliding window.

- CT-FM: 0.2pp gap (cardiac SSL = implicit ROI)
- Random-init: 10–15pp gap (patch OK, full volume over-segments)

---

## Slide 9: Active Jobs

| Job | Description | Status |
|---|---|---|
| ~~8311~~ | Full+BCS corrected test | **DONE** — Dice=0.797, BCS=0.735 |
| ~~8312~~ | ASOCA replicate eval | **Timed out** (k=1: 4/5 done) |
| ~~8308~~ | clDice ASOCA | **DONE** |
| ~~8287_0~~ | HU800-only retrain (full) | **DONE** — best val=0.808 |
| ~~8325~~ | Probe runs (Full+BCS/clDice matched) | **DONE** — peak at L1 (not L2!) |
| **8287_1** | Aug-heavy retrain (TMI) | **Running** (~ep93/100, ~3 days elapsed) |
| **8338** | ASOCA replicate eval v2 (2-day limit) | **Running** (~4h elapsed) |

---

## Slide 10: Paper Status — All Numbers Verified

**All tables filled, all numbers triple-checked against source JSON files.**

**Updates since Mar 7:**
- All numbers verified against bootstrap JSON (n=10,000 resamples)
- Bootstrap significance added to §4.1 text (BCS: p<0.001, clDice Dice: p=0.022)
- Abstract updated: Full+clDice best (Dice~0.801), three-tier CKA
- Contribution item 2 rewritten: "matched LR preserves features (CKA≈0.75)"
- Probe table updated with Full+BCS/clDice matched results (peak at L1, not L2)
- Confounded probe row greyed out with dagger footnote
- Replicate mean±SD added to low-shot table
- Bold markings corrected (clDice Full Dice=0.801 is best, not 0.794)

**Number verification (Mar 9):**
- Ablation table: all 12 rows verified against source JSON — exact match
- Bootstrap significance markers added to table (BCS all p<0.001, Dice varies)
- BCS weight paragraph: fixed false monotonicity claim for partial rows
- CKA table + probe table: all values verified — exact match
- ASOCA zero-shot table: all values verified
- ASOCA low-shot table: all values verified — exact match
- Minor issues identified: CKA≥0.84 should be ≥0.83 (partial w=0.03 = 0.826)

**Remaining before submission (deadline ~April 2):**
- Update replicate error bars when Job 8338 finishes (k=5, k=10)
- Fix CKA≥0.84 → ≥0.83 throughout paper
- Final proofread

---

## Slide 11: New Finding — Probe Peak Shift Was LR Artifact

Old claim: "Full fine-tuning shifts linear probe peak from L1 to L2"
**Corrected**: That was the confounded run (LR=1e-4). At matched LR (1e-5):

| Strategy | L0 | **L1** | L2 | L3 | L4 |
|---|---|---|---|---|---|
| Frozen (any loss) | 0.69 | **0.81** | 0.77 | 0.71 | 0.61 |
| Full+BCS matched | 0.69 | **0.81** | 0.78 | 0.71 | 0.61 |
| Full+clDice matched | 0.69 | **0.82** | 0.78 | 0.71 | 0.63 |
| ~~Confounded (1e-4)~~ | ~~0.68~~ | ~~0.76~~ | ~~**0.80**~~ | ~~0.76~~ | ~~0.69~~ |

All strategies peak at L1 → encoder structure preserved. Only the confounded 10× LR shifts peak to L2.

---

## Slide 12: Three-Tier CKA Structure

| Tier | CKA at bottleneck | Description |
|---|---|---|
| PEFT (frozen/partial/BitFit/adapter) | ≥0.84 | Near-identical to pretrained |
| Full FT at matched LR (1e-5) | ≈0.75 | Moderate adaptation |
| ~~Confounded Full FT (1e-4)~~ | ~~0.37~~ | ~~Pretrained features destroyed~~ |

The adaptation strategy—not the loss function—determines whether pretrained representations survive.

---

## Slide 13: Discussion Points

1. **Full+clDice vs Full+BCS**: clDice wins on Dice (0.801 vs 0.797), BCS wins on BCS (0.735 vs 0.717). Paper narrative: "adaptation strategy > loss choice" — both work well under matched settings.

2. **Replicate variance is large**: k=1 Dice ranges 0.661–0.738 (SD=0.032). Cherry-picked best vs mean is a 4pp gap. Important caveat for low-shot claims.

3. **Fully Soft BCS → ASOCA**: BCS *works* (real gradients) but *hurts* — confirms protocol mismatch. PhD-relevant but not in MIUA paper.

4. **TMI paper**: Architecture + preprocessing results complete (8287_0 done). Start writing?

5. **Stenosis**: When to begin ImageCAS perturbation experiments?

6. **MICCAI reviews**: Expected timeline?
