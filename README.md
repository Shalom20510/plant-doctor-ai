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

The app also screens for low resolution, blur, extreme brightness, unusual framing, and potentially busy backgrounds. Severe quality warnings and ambiguous CLIP image validation ask the user to upload a clearer leaf photo. These checks are heuristic and may make mistakes; they are not a guarantee of disease-diagnosis accuracy.

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
