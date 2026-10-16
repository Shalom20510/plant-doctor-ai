# Plant Doctor AI

A computer-vision prototype that predicts possible plant-leaf health conditions from an uploaded image.

## Current status

The repository includes the Streamlit interface, EfficientNet-B0 model loader, image prediction pipeline, training/evaluation scripts, a trained checkpoint at `artifacts/models/model_v1.pt`, and its class mapping at `artifacts/reports/class_to_idx.json`. The training dataset is not included; provide a compatible labeled dataset to reproduce evaluation and robustness results.

## Supported baseline

The training pipeline expects a folder-based dataset such as the PlantVillage color dataset, with one directory per class:

```text
data/plantvillage/color/
├── Apple___Apple_scab/
│   ├── image1.jpg
│   └── ...
├── Apple___healthy/
└── ...
```

Class names are read from the dataset folders in alphabetical order. The checkpoint stores that exact class order so inference and evaluation can use the same labels.

## Setup (Windows PowerShell)

Use Python 3.10 or 3.11 in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a PyTorch/torchvision build suitable for your computer from the official PyTorch installation selector if the default dependency installation does not match your CPU/GPU setup.

## Train a baseline model

After placing the dataset in the expected folder structure:

```powershell
python -m src.train --data_dir data/plantvillage/color --epochs 5
```

The script saves the best checkpoint to `artifacts/model_v1.pt` and the epoch history to `artifacts/model_v1.history.json` by default. Unless `--no_pretrained` is supplied, torchvision may download ImageNet pretrained weights for transfer learning; this is separate from downloading the plant-disease dataset.

Useful options include `--batch_size`, `--val_fraction`, `--seed`, `--learning_rate`, `--output`, and `--no_pretrained`. Training time depends heavily on dataset size and hardware.

## Evaluate the checkpoint

```powershell
python -m src.eval --data_dir data/plantvillage/color --checkpoint artifacts/models/model_v1.pt
```

Evaluation recreates the deterministic per-class validation split using the same seed and validation fraction as training by default. It prints accuracy, macro-F1, and per-class metrics, and saves a JSON report and confusion matrix data under `artifacts/evaluation/evaluation_report.json`. For a stronger final assessment, use an independent test dataset that was not used during model selection.

## Run the app

```powershell
streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit, upload a JPG, JPEG, or PNG leaf image, and click **Analyze leaf**. The app loads `artifacts/models/model_v1.pt` and `artifacts/reports/class_to_idx.json` from the repository.

## Image robustness evaluation

Place a labeled dataset in the folder-based structure shown above, with the exact class names/order used by the checkpoint, then run:

```cmd
py -m src.robustness --data_dir data/plantvillage/color --checkpoint artifacts/models/model_v1.pt
```

The script compares the original images with synthetic dim/bright, 15°/30° rotation, Gaussian blur, and synthetic edge-clutter variants. It reports accuracy, macro-F1, and accuracy change versus the original images in `artifacts/evaluation/robustness_report.json`. When `--max_images` is used, it selects a deterministic approximately class-balanced subset rather than simply taking the first files (which could overrepresent one class).

For a quick smoke test, use `--max_images 20`; remove that option for the full dataset. This report is only as representative as the dataset. The edge-clutter transformation is a simple synthetic stress test, not a realistic background replacement. For final results, use a separate held-out test set and include real field photos.

The app also screens for low resolution, blur, extreme brightness, unusual framing, and potentially busy backgrounds. Low supported-crop scope scores and severe image-quality warnings can trigger an inconclusive result and ask the user to upload a clearer leaf photo. These checks are heuristic and may make mistakes; they are not a guarantee of disease-diagnosis accuracy.

## Mission 03 — Unsupported input detection and reliable predictions

**Mission 04 evaluation notes:** See [`docs/MISSION_04_EVALUATION.md`](docs/MISSION_04_EVALUATION.md) for the rejection approach, observed manual cases, automated test scope, limitations, and a short demo sequence.

The Streamlit app runs two separate checks before showing a disease result:

1. **Input-scope screening:** CLIP compares the uploaded image against five text descriptions per supported crop, including a morphology-focused description for each of the 14 crop groups (apple, blueberry, cherry, corn, grape, orange, peach, bell pepper, potato, raspberry, soybean, squash, strawberry, and tomato). The app averages the prompt logits within each crop before comparing the best crop against unsupported-leaf and non-leaf categories. Inputs outside the supported scope or below the scope-score threshold are rejected with an explanation.
2. **Disease-model confidence rule:** if the top disease-class softmax score is below `LOW_CONFIDENCE_THRESHOLD = 0.60`, the app shows an **inconclusive** result, lists the closest model categories as alternatives rather than diagnoses, logs the rejection, and does not show disease-care guidance.

For supported-candidate images that pass both checks, the app displays the predicted crop/disease category, model score, alternatives, and general care information. A score of 60% is a prototype decision threshold, not a verified probability that the prediction is correct.

### Automatic crop-consistency rejection

The UI does not ask the user to choose a crop. CLIP screens the image automatically, and the disease classifier predicts across its trained classes. If the crop candidate from CLIP and the crop label from the disease classifier disagree, the app returns an inconclusive mismatch result instead of displaying a disease prediction. This is a heuristic safeguard: it can reject valid images when CLIP is wrong, and matching crop labels do not guarantee a correct diagnosis.

### How to demonstrate the required cases

Run the app and test these separate inputs:

| Case | Suggested input | Expected behavior |
|---|---|---|
| Supported candidate | A sharp, well-lit image of one tomato or apple leaf | Passes the scope screen if CLIP matches a supported crop; disease result appears only if the disease score is at least 60% |
| Non-leaf | A photo of a person, vehicle, food, or building | Rejected as non-leaf; disease classifier is not run |
| Unsupported plant | A leaf from a crop not represented in the 14 supported crop groups | Rejected as unsupported when the unsupported-leaf prompt wins |
| Ambiguous / poor quality | Blurry, dark, distant, or visually unclear leaf | Rejected by the scope or image-quality check, or marked inconclusive |

Use actual photos and record the observed result for your demo. The expected outcomes above describe the intended rules; they are not a claim that every real image has already been tested.

### Dataset-free rule tests

Run:

```cmd
py -m pytest tests/test_input_validation.py -q
```

These tests verify the decision logic for supported candidates, non-leaf inputs, unsupported plant prompts, ambiguous/low-scope inputs, low supported-crop scores, and invalid score values. The live image-screening path compares supported crops, unsupported-leaf prompts, and non-leaf prompts; it relies on low scope scores and image-quality heuristics for ambiguity rather than a separate ambiguity prompt. They use deterministic score examples and **do not** run CLIP on actual photos or establish real-world rejection accuracy.

### Rejection-method limitations

- CLIP scores are relative similarities/softmax scores across prompt groups, not calibrated probabilities of crop identity. Averaging five descriptions may reduce prompt-wording sensitivity, but it has not yet been shown to improve crop accuracy on a labeled test set.
- The morphology descriptions encode general visual cues rather than definitive botanical identification. A tomato-like image can still be mislabeled as soybean or bell pepper; crop-consistency rejection prevents a disease result when CLIP and the disease model disagree, but both can still agree on a wrong crop. The 0.28 scope-score threshold is heuristic and has not been calibrated against a labeled rejection dataset.
- Softmax confidence can be high for a wrong disease class. The 0.60 threshold reduces some low-score predictions but does not guarantee correctness.
- The 14 crop groups are the scope of the checkpoint's 38 disease/healthy classes; not every variety, disease, pest, nutrient problem, or environmental stress is represented.
- Image-quality heuristics and CLIP screening are not a substitute for evaluation on held-out supported, unsupported, non-leaf, and ambiguous images. Report measured acceptance/rejection rates only after running such a labeled test set.

## Important limitations

- This is a prototype, not a definitive plant-health diagnosis.
- Softmax confidence is not a guarantee that a prediction is correct.
- Performance on controlled dataset images may not generalize to field photos, lighting changes, mixed symptoms, or unsupported plant species.
- Confirm important crop-treatment decisions with a qualified agricultural expert.


## Current robustness-test status (2026-10-16)

Implementation is present, but measured robustness results are **pending** because the labeled PlantVillage dataset is not included in this repository and was not found in the local project folders during the latest test. Running the robustness command without that dataset correctly stops with a `FileNotFoundError`; no accuracy or macro-F1 values should be reported for transformed images until the test is actually run.

### Quick verification checklist

- [x] Synthetic transformations implemented: dim/bright, rotations, Gaussian blur, synthetic edge clutter.
- [x] Streamlit image-quality heuristics and uncertainty/clearer-image handling implemented.
- [x] Robustness evaluation script writes per-condition accuracy, macro-F1, and accuracy delta to JSON.
- [ ] Place a compatible labeled dataset in `data/plantvillage/color/<class_name>/*`.
- [ ] Confirm dataset class names match the checkpoint class mapping.
- [ ] Run the robustness evaluation and retain `artifacts/evaluation/robustness_report.json`.
- [ ] Review actual before/after metrics and document findings before making performance claims.

### Run a quick evaluation

From the repository root on Windows:

```cmd
py -m src.robustness --data_dir data/plantvillage/color --checkpoint artifacts/models/model_v1.pt --max_images 50
```

This is a small smoke test, not a full benchmark. Use the same labeled images for each synthetic condition, and use an independent held-out test set for final claims. Do not fabricate missing metrics. The recorded checkpoint validation metrics are not robustness results.


### Dataset-free smoke tests

The transformation helpers have lightweight tests that do not require the PlantVillage dataset:

```cmd
py -m unittest discover -s tests -v
```

These tests check that expected transformations are produced, image dimensions/modes are preserved, synthetic clutter is repeatable for a fixed seed, and brightness variants differ. They do **not** measure classifier accuracy or establish real-world robustness.
