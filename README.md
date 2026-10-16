# Plant Doctor AI

A computer-vision prototype that predicts possible plant-leaf health conditions from an uploaded image.

## Current status

The repository includes the Streamlit interface, EfficientNet-B0 model loader, image prediction pipeline, dataset splitting, training script, and evaluation script. **A trained plant-disease checkpoint and dataset are not included**, so predictions will not work until you prepare data and train a model (or provide a compatible checkpoint).

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
python -m src.eval --data_dir data/plantvillage/color --checkpoint artifacts/model_v1.pt
```

Evaluation recreates the deterministic per-class validation split using the same seed and validation fraction as training by default. It prints accuracy, macro-F1, and per-class metrics, and saves a JSON report and confusion matrix data under `artifacts/evaluation/evaluation_report.json`. For a stronger final assessment, use an independent test dataset that was not used during model selection.

## Run the app

```powershell
streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit, upload a JPG, JPEG, or PNG leaf image, and click **Analyze leaf**. The app expects a compatible trained checkpoint at `artifacts/model_v1.pt` by default; you can enter another checkpoint path in the interface.

## Important limitations

- This is a prototype, not a definitive plant-health diagnosis.
- Softmax confidence is not a guarantee that a prediction is correct.
- Performance on controlled dataset images may not generalize to field photos, lighting changes, mixed symptoms, or unsupported plant species.
- Confirm important crop-treatment decisions with a qualified agricultural expert.
