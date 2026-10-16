# Mission 04 — Input validation and uncertainty evaluation

## What the prototype does

- **Non-leaf screening:** CLIP compares the uploaded image with person, building, vehicle, food, animal, and unrelated-object prompts. If a non-leaf category wins, the disease classifier is not run.
- **Unsupported plant screening:** CLIP compares supported crop prompts with prompts for unsupported/unknown plant leaves. If the unsupported-leaf category wins, the app returns an unsupported result.
- **Ambiguous image screening:** CLIP now also compares an explicit unclear/blurry/ambiguous-image prompt. Basic image-quality checks flag very small images, unusual framing, blur, extreme brightness, and busy backgrounds. These are heuristics, not a trained quality classifier.
- **Disease confidence rejection:** the app rejects predictions below `LOW_CONFIDENCE_THRESHOLD = 0.60` and shows an inconclusive result rather than disease guidance.
- **Crop consistency:** in automatic mode, if the CLIP crop and disease model crop disagree, the app returns an inconclusive result. The user can select a known crop; inference is then limited to that crop's available classes.
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
- rejection when the ambiguous prompt wins;
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
- A model result of “healthy” must not be treated as proof that a plant is disease-free.
- The training dataset is not present in this repository, so a full held-out evaluation, threshold calibration, and retraining/fine-tuning on real field photos have not been performed.
- Image-quality checks and CLIP screening can both falsely accept or reject images. Users should take a clear, well-lit close-up of one leaf and seek expert confirmation before making treatment decisions.

## Recommended demo sequence

1. Upload the known photo of people and show the **Non Leaf** rejection.
2. Upload a clear photo of a supported crop; choose its crop in the selector and show the crop-conditioned model output. State that this is a model prediction, not ground-truth validation, unless independently verified.
3. Upload a blurry or unclear leaf and show an inconclusive result if the quality/scope checks reject it.
4. Explain that the prototype blocks some unreliable inputs but is not yet validated for reliable field diagnosis.
