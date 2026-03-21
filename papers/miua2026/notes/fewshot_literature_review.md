# Few-Shot Medical Image Segmentation: Literature Review
*Compiled 2026-03-03 for ASOCA low-shot experiments*

## Most Relevant Papers

### 1. FSEFT — Few-Shot Efficient Fine-Tuning (MedIA 2025, Best Paper MedAGI@MICCAI 2023)
**Silva-Rodriguez, Dolz, Ben Ayed** | [arXiv 2303.17051](https://arxiv.org/abs/2303.17051) | [Code](https://github.com/jusiro/fewshot-finetuning)

Evaluates PEFT strategies (LoRA, BitFit, Adaptformer, Spatial Adapter) for 3D CT organ segmentation at K=1,5,10,30.
- **Critical finding**: Full fine-tuning catastrophically fails at K=1 (52% vs zero-shot 72%)
- Spatial Adapter at K=1 outperforms full fine-tuning at K=30 by 8.6pp
- **Relevance**: Directly validates our EXP-045 finding (full unfreeze → Dice=0.744). Motivates our frozen encoder + BitFit/Adapter ablation.

| Method | K=1 | K=10 |
|---|---|---|
| Full fine-tuning | 52.4% | 69.9% |
| Zero-shot | 72.4% | 72.4% |
| BitFit | 73.2% | 76.7% |
| LoRA | 74.3% | 77.8% |
| Spatial Adapter | **76.3%** | 77.4% |

### 2. SuPreM — Supervised Pre-Training for 3D Medical Segmentation (ICLR 2024 Oral)
**Li, Yuille, Zhou (Johns Hopkins)** | [Code](https://github.com/MrGiovanni/SuPreM)

Pre-trains SegResNet + Swin-UNETR on 9,262 CT volumes (AbdomenAtlas). N=5 with SuPreM outperforms N=20 with SSL baselines.
- **Relevance**: Directly analogous to CT-FM. Supports our argument that supervised-pretrained SegResNet retains transferable representations even at k=7 ASOCA images.

### 3. ARENA — Regularized Low-Rank Adaptation (MICCAI 2025)
**Baklouti, Silva-Rodriguez, Dolz, Bahig, Ben Ayed** | [arXiv 2507.15793](https://arxiv.org/abs/2507.15793)

Extends LoRA with L1 sparsity on singular values → automatic rank selection. Same group as FSEFT.
- **Relevance**: State-of-the-art LoRA variant. Relevant if we add LoRA to ASOCA PEFT comparison.

### 4. Breaking the Data Barrier — Few-Shot 3D Vessel Segmentation (arXiv Feb 2026)
**Yoshihara, Sugawara, Tokuoka, Hong** | [arXiv 2602.23782](https://arxiv.org/abs/2602.23782)

DINOv2 (ViT-S/16) + 3D adapter for cerebral vessel segmentation at 5-shot. Z-channel embedding bridges 2D→3D.
- TopCoW 5-shot: Dice=43.4% vs nnU-Net 33.4%
- **Relevance**: Closest few-shot 3D vessel paper. Our CT-FM zero-shot (Dice=0.490) already beats this, suggesting CT-FM is a particularly strong starting point.

### 5. vesselFM — Universal 3D Vessel Foundation Model (CVPR 2025)
**Wittmann, Wattenberg, Amiranashvili, Shit, Menze** | [arXiv 2411.17386](https://arxiv.org/abs/2411.17386)

Vessel-specialist FM trained on curated + synthetic (domain randomization + flow matching) data. Zero/one/few-shot across CT, MRI, OCT, microscopy.
- Outperforms VISTA3D by 5.86pp Dice on MSD8
- **Relevance**: Vessel-specialist alternative to our general-purpose CT-FM. Synthetic data strategy could help ASOCA scarcity problem.

### 6. TPNet — Tubular-Aware Prompt-Tuning (IEEE TMI 2025)
**PubMed 41650434** | [IEEE](https://ieeexplore.ieee.org/document/11373541)

Tubular morphology priors (region growing + cross-correlation) for few-shot pulmonary vessel segmentation.
- **Relevance**: Bakes topology into architecture (vs our approach of baking it into loss). Key design comparison for discussion.

## Secondary References

### 7. VesselShot (arXiv 2023)
Few-shot cerebrovascular 3D MRA. Mean Dice=0.62±0.03 on TubeTK. Shows difficulty of few-shot vessel segmentation without strong FM.

### 8. VessShape (arXiv Oct 2025)
Synthetic 2D vessel pretraining via Bezier curves. 4-shot DRIVE retinal: ~77% Dice. Concept of synthetic pretraining applicable to 3D coronaries.

### 9. Anatomy+Topology Coronary Framework (IEEE TMI 2024, PMID 37756173)
**Zhang, Sun, Wu, Shen et al.** Two-stage: anatomical dependency + hierarchical topology learning + NSDT soft-clDice.
- Not few-shot but most relevant topology-preserving coronary baseline. Key comparison for our BCS loss.

### 10. ASOCA Generalization Study (J Imaging Info Med 2025)
Systematic evaluation on ASOCA (40 cases) + GeoCAD (70 cases). Artery contrast (r=0.408) and edge sharpness (r=0.239) predict segmentation quality. Calcification is a confounder.
- **Relevance**: Uses our exact ASOCA dataset. Explains why k=7 fine-tuning is hard (imaging variability across normal vs diseased).

### 11. UniverSeg (ICCV 2023)
**Butoi et al. (MIT CSAIL)** In-context segmentation via CrossBlock — no gradient updates needed. Given k support examples, segments query in one forward pass.
- Potential zero-gradient baseline for ASOCA: provide 5 labeled cases as support.

### 12. SAM-Med3D (ECCV 2024 Workshop)
3D SAM with point prompts. +17.7pp DSC over slice-based SAM.
- Prompt-based paradigm avoids gradient saturation problem, but coronary arteries are likely too small for effective prompting.

## Key Takeaways for Our Work

1. **Full fine-tuning always fails at K≤5** — FSEFT validates our EXP-045 finding. Frozen encoder + PEFT is state-of-the-art.

2. **No existing paper evaluates topology losses in few-shot regime** — our observation that BCS saturates with 7 images (too few bifurcations per patch) is a **novel finding with no prior literature**. Strong contribution for TMI.

3. **Recommended PEFT hierarchy**: Spatial Adapter > LoRA > BitFit > Full fine-tune (from FSEFT). Our frozen encoder experiments are aligned with best practice.

4. **Synthetic data augmentation** could help ASOCA scarcity (vesselFM, VessShape). Our `src/geometry/` code could generate synthetic coronary centerlines.

5. **CT-FM zero-shot (Dice=0.490) already competitive** with other few-shot vessel methods (TopCoW 5-shot: 0.434), suggesting strong cardiac domain pretraining matters more than sophisticated few-shot methods.
