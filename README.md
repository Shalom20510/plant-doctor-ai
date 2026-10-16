# Plant Doctor AI

A computer-vision project to help identify possible plant leaf diseases from an uploaded image.

## Project status

**Foundation stage:** the repository structure and configuration are being established. Prediction, model training, and evaluation will be implemented in later steps.

## Planned features

- Upload a plant-leaf image.
- Predict a supported plant/disease class using a trained image-classification model.
- Display the prediction and confidence with appropriate limitations.
- Keep data loading, model code, and prediction logic separated.
- Add tests and usage documentation.

## Project structure

```text
app/                 Streamlit interface
src/                 Model, data, and prediction modules
configs/             Application configuration
tests/               Automated tests
requirements.txt     Python dependencies
```

## Setup

Use Python 3.10 or 3.11 in a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The application is not runnable yet; its interface and inference pipeline will be implemented in subsequent steps.

## Important limitation

A model prediction is an aid, not a definitive plant-health diagnosis. Results depend on image quality, supported classes, and the data used to train and evaluate the model. Confirm important crop-treatment decisions with a qualified agricultural expert.
