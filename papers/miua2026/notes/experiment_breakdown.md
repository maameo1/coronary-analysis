# MIUA Paper — Experiment Breakdown (Plain English)

## What's the paper about?

We have a pretrained AI model (CT-FM) that was trained on 148K CT scans to understand anatomy. We want to fine-tune it to segment coronary arteries — specifically, we want it to get the **branching structure** right, not just the volume. The paper asks two questions:

1. **How much of the model should we let change?** (freeze the encoder, or let it adapt?)
2. **Does this work on a completely different dataset?** (ASOCA — different hospital, different annotation style)

---

## Part 1: The Ablation Grid (ImageCAS)

### What is the grid?

We cross **5 adaptation strategies** with **3 BCS loss weights**, giving us a matrix of experiments. Each experiment starts from the same L1 checkpoint (the model after vesselness training — it already knows what vessels look like, but doesn't yet care about branching).

### The 5 adaptation strategies (HOW MUCH do we change?)

**1. Frozen (decoder-only)** — EXP-044, EXP-048, EXP-046
- **What**: Lock the encoder completely. Only the decoder (the part that produces the final segmentation) can learn.
- **Why**: The encoder learned good features from 148K CTs. Maybe those features are already enough for topology — the decoder just needs to learn to use them.
- **Params trainable**: 10.0M out of 87.8M (11%)
- **Analogy**: Like having a fluent translator (encoder) and only training a new secretary (decoder) to format their output differently. The translator doesn't change.

**2. BitFit** — EXP-052
- **What**: Lock everything EXCEPT the batch normalisation bias terms in the encoder (14K tiny params). Decoder is fully trainable.
- **Why**: BitFit is a known trick from NLP — just tweaking biases can shift feature distributions without rewriting features. Tests whether even tiny encoder changes help.
- **Params trainable**: 10.0M + 14K (same as frozen, essentially)
- **Analogy**: Like adjusting the volume knobs on a mixing board without re-recording any tracks.

**3. Adapters** — EXP-053
- **What**: Insert small trainable bottleneck modules (175K params) at skip connections between encoder and decoder. Encoder itself is frozen.
- **Why**: Adapters are a popular parameter-efficient method — they let the model transform encoder features before the decoder sees them, without touching the encoder weights.
- **Params trainable**: 10.2M (10.0M decoder + 175K adapters)
- **Analogy**: Like putting a filter on the translator's output before the secretary sees it.

**4. Partial unfreeze** — EXP-049, EXP-047, EXP-050
- **What**: Unfreeze the last encoder block (the deepest/most abstract features) plus the full decoder. Earlier encoder layers stay frozen.
- **Why**: Maybe the deepest encoder features need to shift to support topology. This is a middle ground between frozen and full.
- **Params trainable**: 27.7M (32%)
- **Analogy**: Like letting the translator revise their summary but not re-translate the whole document.

**5. Full fine-tuning** — EXP-045
- **What**: Everything is trainable. No freezing at all.
- **Why**: Negative control. If the model can change everything, it has the most freedom to learn topology — but also the most freedom to overfit.
- **Params trainable**: 87.8M (100%)
- **Analogy**: Like rewriting the whole document from scratch. Maximum flexibility, but you might lose what was good about the original.

### The 3 BCS weights (HOW HARD do we push topology?)

- **λ=0.03**: Gentle topology nudge. The model mostly optimises for Dice/CE (volume overlap), with a small BCS correction.
- **λ=0.05**: Moderate. Balanced push for both overlap and topology.
- **λ=0.10**: Strong topology push. The model is heavily penalised for missing branch connections.

**How BCS loss works**: At each bifurcation (branching point) in the training patch, it checks: "did you segment ALL branches, or did you miss one?" It focuses on the weakest branch — if one branch is missed, the whole bifurcation scores badly. The loss is: "1 minus the average bifurcation score."

### What we found

| Strategy | Best Dice | Best BCS | Takeaway |
|---|---|---|---|
| Frozen λ=0.03 | **0.793** | 0.716 | Best Dice in the whole grid |
| Frozen λ=0.05 | 0.792 | **0.722** | Best Dice-BCS balance |
| Frozen λ=0.10 | 0.785 | **0.730** | Best BCS but Dice drops |
| BitFit λ=0.05 | ~0.793 | ~0.720 | Essentially same as frozen — bias tweaks don't help |
| Adapters λ=0.05 | ~0.793 | ~0.720 | Essentially same as frozen — skip adapters don't help |
| Partial λ=0.05 | 0.792 | 0.724 | Barely different from frozen |
| Full λ=0.10 | 0.748 | 0.739 | Highest BCS... but Dice drops 4.5pp (catastrophic overfitting) |

**Key insight**: Frozen wins. The encoder already contains topology-relevant features — you don't need to change it. Changing it either doesn't help (BitFit, adapters, partial) or actively hurts (full). The decoder alone can learn to exploit the topology signal.

**CKA analysis** confirms this: frozen/partial/BitFit/adapters all have CKA ≥ 0.84 with the original encoder (features barely changed). Full drops to 0.37 at the bottleneck (features completely rewritten → overfitting).

---

## Part 2: ASOCA Cross-Domain Transfer

### What's the problem?

ASOCA is a different coronary artery dataset (40 cases, different hospital). Critical difference: **ASOCA only labels major vessels** (the big trunks). ImageCAS labels **the full tree** (including small side branches).

So when our ImageCAS-trained model sees an ASOCA scan, it correctly segments small branches — but ASOCA's ground truth says those branches don't exist. The model gets penalised for being right.

### Experiment 1: Zero-shot (no fine-tuning)

**What**: Just run all our ImageCAS models on ASOCA without any adaptation.
**Why**: Baseline — how well does it transfer out-of-the-box?

**Result**: Dice=0.481, BCS=0.794. Terrible Dice (over-segmentation), but great BCS (it's finding all the branches). BCS-trained models have even lower Dice (0.423-0.461) because BCS training made them MORE aggressive at finding branches — exactly what ASOCA penalises.

**Insight**: L1 (no BCS training) actually beats ALL BCS models on ASOCA Dice. BCS training amplifies the annotation mismatch — the model is confidently segmenting real vessels that ASOCA doesn't label.

### Experiment 2: Threshold calibration

**What**: Instead of using the default 0.5 probability threshold to binarise predictions, try higher thresholds (0.6, 0.7, 0.8, 0.9).
**Why**: The model is over-confident — it assigns high probability to thin branches that ASOCA doesn't want. Raising the threshold means "only keep the vessels you're REALLY sure about."

**Result**: Threshold 0.9 recovers +18.6pp Dice (0.481→0.665) with only -1.6pp BCS loss. Free improvement — no retraining needed, just change one number at inference time.

**Insight**: BCS stays remarkably stable across thresholds (±2pp). The branches the model is confident about are genuinely connected — changing the threshold removes thin extensions but doesn't break topology.

### Experiment 3: Low-shot fine-tuning

**What**: Fine-tune on tiny amounts of ASOCA data (k=1, 5, 10 training cases) with shallow decoder freezing (only last 2 decoder levels trainable).
**Why**: Can a tiny amount of target data teach the model ASOCA's annotation style without destroying what it learned on ImageCAS?

**Results**:

| Training data | Dice | BCS | HD95 | Betti |
|---|---|---|---|---|
| Zero-shot (no training) | 0.481 | 0.794 | 15.7 | 7.1 |
| k=1 (2 scans), t=0.7 | **0.706** | **0.752** | **10.6** | 13.1 |
| k=5 (10 scans), t=0.4 | 0.695 | 0.695 | 26.8 | 46.7 |
| k=10 (20 scans), t=0.4 | 0.671 | 0.751 | 46.4 | 64.6 |

**The paradox**: More data makes it WORSE. k=1 is the best, k=10 is the worst.

**Why**: With just 2 training scans, the model makes a small correction ("oh, ASOCA doesn't want side branches"). With 10-20 scans, it overfits to ASOCA's sparse annotation style, losing the rich topology features from ImageCAS. The Betti error explodes (7.1→64.6) meaning the model creates tons of disconnected fragments.

### Experiment 4: Comparison with full decoder fine-tuning (EXP-054b)

**What**: Our earlier attempt — fine-tune the entire decoder (not just 2 levels) with k=7 training cases and aggressive BCS weight (0.30).
**Why**: To show the shallow freeze matters.

**Result**: Dice=0.701, BCS=0.679, Betti=36.4. Similar Dice to k=1 shallow, but BCS drops 7pp and Betti is 3× worse. Full decoder fine-tuning erases the pretrained topology features.

**Insight**: Shallow decoder freeze (depth=2) is crucial. It lets the model learn ASOCA's annotation style while preserving the deeper topology-aware representations.

### Experiment 5: Starting from BCS checkpoint (EXP-048 → ASOCA)

**What**: Instead of fine-tuning from L1 (no BCS), fine-tune from EXP-048 (frozen w=0.05, best BCS on ImageCAS).
**Why**: Maybe starting with a model that already understands topology gives a better BCS floor after adaptation.

**Status**: Running (job 7839). Hypothesis: BCS-pretrained start should maintain higher BCS during adaptation.

---

## Part 3: Representational Analysis

### CKA (Centered Kernel Alignment)

**What**: Measures how similar the internal representations (feature maps) are between two models at each layer.
**Why**: Answers "did fine-tuning actually change what the encoder sees, or just how the decoder uses it?"
**How**: Feed same images through L1 (base) and fine-tuned model, extract features at each encoder layer, compute CKA similarity (1.0 = identical, 0.0 = completely different).

**Finding**: Frozen/BitFit/adapters: CKA ≥ 0.84 everywhere. The encoder didn't change. Full fine-tuning: CKA drops to 0.37 at the bottleneck — the model rewrote its deepest features, which is why it overfits.

### Linear Probe (Encoder)

**What**: Train a tiny classifier at each encoder layer to predict "is this voxel near a bifurcation?"
**Why**: Tests whether topology information exists in the encoder at each depth level.
**How**: Freeze everything, attach a small linear layer at each encoder level, train only that layer.

**Finding**: Peak AUC at level 1 (~0.81). The encoder already encodes bifurcation information in its shallow features — no fine-tuning needed.

### Decoder Probe

**What**: Same as linear probe but at each decoder level.
**Why**: Shows where the decoder uses the topology signal.

**Finding**: Peak at decoder level 2 (~0.76). This is where bifurcation information gets decoded into the final segmentation.

---

## Part 4: Supervisor Feedback Experiments (Feb 27)

### clDice Comparison (Job 8077)

**What**: Run the same frozen vs full comparison but with clDice loss instead of BCS.
**Why**: Supervisor said the conclusion is too strong with only one topology loss. "We could do clDice though." If frozen+clDice also works, it strengthens the claim that decoder-only fine-tuning works for topology objectives in general, not just BCS.

| Experiment | Loss | Freeze | Compare to |
|---|---|---|---|
| clDice-frozen | clDice w=0.05 | Frozen | EXP-048 (BCS, frozen) |
| clDice-full | clDice w=0.10 | Full | EXP-045 (BCS, full) |

**Expected outcome**: If frozen+clDice preserves Dice while full+clDice overfits, we've shown the conclusion holds across topology losses.

### Multi-Model Ensemble (Job 8065)

**What**: Average predictions from 4 models (L1 + EXP-044 + EXP-048 + EXP-047).
**Why**: Supervisor concerned about gap to SOTA (~0.793 vs ~0.835 Dice on ImageCAS). Ensembling diverse models might close it.
**Status**: Running, 40/250 metrics computed. Early results: Dice=0.787, HD95=6.36 (HD95 looks great).

---

## The Paper's Story (in one paragraph)

We have a powerful CT foundation model. When we add topology-aware supervision (BCS loss) to teach it about vessel branching, the critical question is: how much of the model should we let change? Answer: **only the decoder**. The encoder already knows about topology from pretraining — the decoder just needs to learn to exploit it. Changing the encoder (BitFit, adapters) doesn't help, and fully changing it destroys generalisation. On a different dataset (ASOCA), the model over-segments because it correctly finds branches that ASOCA doesn't label. The fix is simple: raise the threshold or fine-tune with just 2 examples — more data paradoxically makes it worse by overfitting to ASOCA's sparse annotations.
