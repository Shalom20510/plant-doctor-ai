# Mission 04 — Input validation and uncertainty evaluation

## What the prototype does

- **Non-leaf screening:** CLIP compares the uploaded image with person, building, vehicle, food, animal, and unrelated-object prompts. If a non-leaf category wins, the disease classifier is not run.
- **Unsupported plant screening:** CLIP compares supported crop prompts with prompts for unsupported/unknown plant leaves. If the unsupported-leaf category wins, the app returns an unsupported result.
- **Ambiguous image screening:** The live CLIP screen rejects low supported-crop scope scores, while basic image-quality checks flag very small images, unusual framing, blur, extreme brightness, and busy backgrounds. The live screen does not currently compare a separate ambiguity prompt. These are heuristics, not a trained quality classifier.
- **Disease confidence rejection:** the app rejects predictions below `LOW_CONFIDENCE_THRESHOLD = 0.60` and shows an inconclusive result rather than disease guidance.
- **Crop consistency:** the UI uses automatic screening only. If the CLIP crop and disease-model crop disagree, the app returns an inconclusive result rather than displaying a disease prediction. There is no manual crop selector.
- **Communication:** rejected inputs display a reason, state that disease prediction was not run, and explain that CLIP scores are relative rather than calibrated probabilities.

## Evaluation evidence

### Automated rule tests

Run from the repository root:

```cmd
py -m pytest -q
```

The dataset-free tests cover:
- a supported-crop candidate passing the scope rule;
- non-leaf rejection;
- unsupported-plant rejection;
- ambiguous/low-scope rejection;
- low disease-model score rejection through the app's confidence rule;
- aggregation of multiple crop prompts;
- ambiguous/low-scope rejection rules;
- crop-conditioned prediction not returning labels from another crop.

These are **unit/integration tests using deterministic inputs or mocked model outputs**. They verify software behavior; they do not measure real-world plant-identification accuracy.

### Manual observations recorded during development

| Input | Observed behavior | Interpretation |
|---|---|---|
| Photo of people | Rejected as **Non Leaf**; CLIP category score was reported as 99.9% | One observed rejection case. The displayed score is relative to the prompt categories, not a calibrated probability. |
| Tomato-like close-up leaf | Disease classifier returned **Bell Pepper — Healthy** at 61.58% | A misleading crop/condition output. This is evidence of a real limitation, not a successful correct diagnosis. |
| Same tomato-like leaf in automatic screening | CLIP crop candidate was **Soybean** while the disease classifier predicted **Bell Pepper**; the app returned **Inconclusive result** and did not report a disease | The crop-mismatch safeguard worked for this observed case, although the crop screen itself was wrong. |

### Cases still requiring real-photo confirmation

A successful, independently verified disease prediction on a supported crop, a real unsupported-species rejection, and a real ambiguous-leaf rejection have **not yet been established with a labeled image set**. Do not describe the deterministic tests as proof of those real-world outcomes. For the live demonstration, use one known supported-crop image and record the exact selected crop, displayed category, score, and whether a knowledgeable person verified the label. Also demonstrate the people-photo rejection above.

## Limits and interpretation

- CLIP's crop score is a relative similarity score and is not a calibrated probability of crop identity.
- The 0.28 scope threshold and 0.60 disease-score threshold are heuristics; neither has been calibrated on a labeled rejection dataset.
- Softmax scores can be high even for a wrong class. Selecting a crop limits the candidate disease classes but does not verify that the user selected the correct plant or guarantee the condition is correct.
- A model result of “healthy” must not be treated as proof that a plant is disease-free. The app no longer asks users to select a crop, because that would shift plant identification to the user instead of demonstrating automatic screening.
- The training dataset is not present in this repository, so a full held-out evaluation, threshold calibration, and retraining/fine-tuning on real field photos have not been performed.
- Image-quality checks and CLIP screening can both falsely accept or reject images. Users should take a clear, well-lit close-up of one leaf and seek expert confirmation before making treatment decisions.

## Recommended demo sequence

1. Upload the known photo of people and show the **Non Leaf** rejection.
2. Upload a clear photo of a supported crop and show the automatic model output. State that this is a model prediction, not ground-truth validation, unless independently verified.
3. Upload a blurry or unclear leaf and show an inconclusive result if the quality/scope checks reject it.
4. Explain that the prototype blocks some unreliable inputs but is not yet validated for reliable field diagnosis.


## Mission 04 submission matrix — expected vs observed

| Test | Input / method | Expected result | Actual evidence currently available | Status |
|---|---|---|---|---|
| Test 1A — supported held-out leaf | Labeled held-out image from one of the 38 model classes | Predicted class plus confidence/uncertainty indicator | A tomato-like leaf previously produced `Bell Pepper — Healthy` at 61.58%; its true label was not independently established. This is not a verified correct prediction. | **INCOMPLETE** — run labeled evaluation |
| Test 1B — aggregate metrics | Independent labeled held-out set | Report macro-F1 and balanced accuracy, plus sample count | `src/eval.py` reports accuracy and macro-F1 on a deterministic validation split. No generated metrics JSON is committed; the dataset is absent. | **INCOMPLETE** — run evaluation |
| Test 2 — changed image conditions | Same held-out images with dim/bright lighting, 15°/30° rotation, Gaussian blur, and synthetic edge clutter | Compare original and transformed accuracy/macro-F1 and report degradation | `src/robustness.py` implements these transformations and writes a JSON report, but measured output is not available. | **INCOMPLETE** — run robustness evaluation |
| Test 3A — non-leaf | Photo of people | Reject as non-leaf; do not run disease classifier | One manual photo-of-people test was rejected as **Non Leaf**; displayed CLIP category score was 99.9% (relative score, not calibrated probability). | **OBSERVED PASS** for this one image |
| Test 3B — unsupported plant | Labeled leaf from a plant outside the supported crop groups | Unsupported/inconclusive result | No verified labeled real-photo result available. | **INCOMPLETE** |
| Test 3C — ambiguous/heavily blurred | Blurred or unclear leaf | Ask for a clearer image or return inconclusive | Rule tests cover deterministic low-scope/ambiguity conditions; no documented measured real-photo result available. | **PARTIAL** — real-photo evidence needed |
| Test 3D — crop disagreement | Tomato-like leaf for which CLIP and classifier predicted different crops | Return inconclusive instead of a disease diagnosis | CLIP chose Soybean while classifier chose Bell Pepper; the app rejected the disease result. | **OBSERVED PASS** for this one image |

## How to generate the missing measured evidence

First, place the labeled dataset in `data/plantvillage/color/<class-name>/*`. The class folder names and alphabetical order must match the checkpoint. Do not report invented metrics or use screenshots of predictions as a substitute for aggregate metrics.

From the repository root on Windows, run:

```cmd
py -m pytest -q
py -m src.eval --data_dir data/plantvillage/color --checkpoint artifacts/models/model_v1.pt
py -m src.robustness --data_dir data/plantvillage/color --checkpoint artifacts/models/model_v1.pt --max_images 140
```

The evaluation command writes `artifacts/evaluation/evaluation_report.json`. The robustness command writes `artifacts/evaluation/robustness_report.json`. The `--max_images 140` run is a quick class-balanced stress test, not a full benchmark; remove the option for the complete dataset.

**Metric interpretation:** macro-F1 is the unweighted mean of per-class F1 scores. Balanced accuracy is the mean per-class recall. The current `src/eval.py` emits accuracy and macro-F1, but not balanced accuracy; do not label accuracy as balanced accuracy. Use an independent held-out set for final claims, because a validation split used during model development is weaker evidence. Record the number of evaluated images, dataset/split, checkpoint, seed, and exact command alongside the metrics.

For transformed-image robustness, report each condition's accuracy and macro-F1 beside the original condition. Accuracy degradation in percentage points is `(transformed_accuracy - original_accuracy) × 100`; negative values indicate worse performance. Synthetic transformations are stress tests, not a substitute for real field photographs.

## Evidence to capture for submission

1. Screenshot of a supported labeled leaf, showing the predicted class and score, with its actual dataset label recorded separately.
2. Terminal output and the saved JSON report for macro-F1 / accuracy (and balanced accuracy if added to the evaluator).
3. Terminal output and JSON table comparing original and transformed-image metrics.
4. Screenshot of a non-leaf rejection and an unsupported or ambiguous/blurred input rejection.
5. Keep failed and incorrect predictions in the report. Never describe a high confidence score as proof that a prediction is correct.

**Submission honesty note:** until the dataset-backed commands have run and the generated reports have been inspected, the aggregate classification and robustness metrics remain pending. The repository contains evaluation tooling and limited manual observations, not completed quantitative validation.
