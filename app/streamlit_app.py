
import os
import sys
from pathlib import Path
from datetime import datetime

import streamlit as st
import torch
from PIL import Image, ImageFilter, ImageStat
from transformers import CLIPModel, CLIPProcessor

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

LOG_FILE = ROOT_DIR / "logs" / "predictions.log"

from src.predict import (
    get_device,
    load_class_mapping,
    load_model,
    get_inference_transform,
    predict_topk,
)

st.set_page_config(
    page_title="Plant Doctor AI",
    page_icon="🌿",
    layout="centered",
)

LOW_CONFIDENCE_THRESHOLD = 0.60

@st.cache_resource
def load_clip_validator():
    model_name = "openai/clip-vit-base-patch32"

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()

    return processor, model


def validate_plant_image(image):
    processor, model = load_clip_validator()

    descriptions = [
        "a close-up photograph of a plant leaf",
        "a photograph of a plant with visible leaves",
        "a photograph of people",
        "a photograph of a building",
        "a photograph of a vehicle",
        "a photograph of food",
        "a photograph of an animal",
        "a photograph of an unrelated object",
    ]

    inputs = processor(
        text=descriptions,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits_per_image.softmax(dim=1)[0]

    best_index = int(scores.argmax().item())
    best_description = descriptions[best_index]
    best_score = float(scores[best_index].item())

    plant_indices = [0, 1]
    plant_score = sum(float(scores[i].item()) for i in plant_indices)

    return best_description, best_score, plant_score



@st.cache_resource
def load_artifacts():
    weights_path = ROOT_DIR / "artifacts" / "models" / "model_v1.pt"
    class_map_path = ROOT_DIR / "artifacts" / "reports" / "class_to_idx.json"

    if not weights_path.exists():
        raise FileNotFoundError(f"Missing model weights: {weights_path}")

    if not class_map_path.exists():
        raise FileNotFoundError(f"Missing class map: {class_map_path}")

    device = get_device()
    class_to_idx, idx_to_class = load_class_mapping(str(class_map_path))

    model = load_model(
        weights_path=str(weights_path),
        num_classes=len(class_to_idx),
        device=device,
    )

    transform = get_inference_transform(img_size=224)

    return model, idx_to_class, transform, device, weights_path.stem


def readable_name(label):
    """Convert a PlantVillage class label into readable text."""
    plant, _, disease = label.partition("___")
    plant = plant.replace("_", " ").replace(",", "").strip()
    disease = disease.replace("_", " ").replace(",", "").strip()
    disease = " ".join(disease.split())

    plant_names = {
        "Corn (maize)": "Corn",
        "Pepper bell": "Bell Pepper",
        "Cherry (including sour)": "Cherry",
        "Orange": "Orange",
    }

    plant = plant_names.get(plant, plant)

    if disease.lower() == "healthy":
        return plant, "Healthy", f"{plant} — Healthy"

    disease = disease.replace("Haunglongbing", "Huanglongbing")
    disease = disease.replace("Gray leaf spot", "Gray Leaf Spot")
    disease = disease.replace("Two spotted spider mite", "Two-Spotted Spider Mite")
    disease = disease.replace("Black measles", "Black Measles")

    disease = " ".join(word.capitalize() for word in disease.split())

    return plant, disease, f"{plant} — {disease}"


def disease_guidance(label):
    """Provide general educational guidance for the 38 model classes."""
    plant, disease, _ = readable_name(label)

    if disease == "Healthy":
        return {
            "status": "Healthy-class prediction",
            "summary": f"The model classified this image as a healthy {plant.lower()} leaf.",
            "symptoms": "This class represents leaves labelled healthy in the training dataset.",
            "action": "Continue regular watering, appropriate nutrition, and routine inspection.",
            "prevention": "Monitor new growth and check both sides of leaves regularly.",
        }

    name = disease.lower()

    if "spider mite" in name:
        category = "pest"
        summary = "This class represents spider-mite damage."
        symptoms = "Possible signs include fine stippling, discoloration, and sometimes webbing."
        action = "Inspect leaf undersides for mites. Confirm the pest before selecting a suitable control."
        prevention = "Monitor plants regularly and manage dust and plant stress."

    elif "virus" in name or "yellow leaf curl" in name or "mosaic" in name:
        category = "possible viral disease"
        summary = "This class represents symptoms associated with a viral disease."
        symptoms = "Possible signs include mottling, curling, yellowing, or distorted growth."
        action = "Check for insect vectors and seek local agricultural advice. There is generally no direct cure for an established plant virus."
        prevention = "Use clean planting material, manage vectors, and follow local sanitation guidance."

    elif "bacterial" in name:
        category = "possible bacterial disease"
        summary = "This class represents a disease associated with bacterial symptoms."
        symptoms = "Possible signs include leaf spots, yellow margins, or water-soaked lesions."
        action = "Avoid handling wet plants, limit overhead watering, and confirm the cause before treatment."
        prevention = "Use clean tools and planting material and remove affected material according to local guidance."

    elif "healthy" in name:
        category = "plant health"
        summary = "The model returned a healthy-class label."
        symptoms = "Compare the whole plant with its normal appearance."
        action = "Inspect the plant and monitor changes over time."
        prevention = "Maintain appropriate growing conditions."

    else:
        category = "possible fungal or oomycete disease"
        summary = "This class represents a leaf disease commonly associated with fungal or fungus-like pathogens."
        symptoms = "Possible signs include spots, lesions, blight, mildew, or areas of dead tissue."
        action = "Remove severely affected fallen material where appropriate, keep foliage dry when practical, and confirm the disease before using any pesticide."
        prevention = "Improve airflow, avoid unnecessary leaf wetness, and use disease-resistant varieties when available."

    return {
        "status": category.title(),
        "summary": summary,
        "symptoms": symptoms,
        "action": action,
        "prevention": prevention,
    }


def image_quality_checks(image):
    """Basic screening heuristics; not a validated image-quality classifier."""
    warnings = []

    width, height = image.size

    if min(width, height) < 160:
        warnings.append(
            "This image has low resolution. Try a larger, clearer photograph."
        )

    gray = image.convert("L")
    edge_image = gray.filter(ImageFilter.FIND_EDGES)
    edge_variance = ImageStat.Stat(edge_image).var[0]

    # Heuristic only. This threshold has not been calibrated on a test set.
    if edge_variance < 100:
        warnings.append(
            "The image may be blurry or lack detail. Try taking a sharper photo "
            "with the leaf in focus."
        )

    brightness = ImageStat.Stat(gray).mean[0]

    if brightness < 35:
        warnings.append("The image is very dark. Try taking the photo in better lighting.")
    elif brightness > 235:
        warnings.append("The image may be overexposed. Avoid direct glare.")

    return warnings


def log_prediction(model_version, image_name, label, probability, status):
    os.makedirs(LOG_FILE.parent, exist_ok=True)

    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", encoding="utf-8") as file:
            file.write(
                "timestamp | model_version | image_name | predicted_class "
                "| confidence | status\n"
            )

    safe_image_name = Path(image_name).name.replace("|", "_").replace("\n", "_")
    safe_label = label.replace("|", "_").replace("\n", "_")

    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"{model_version} | {safe_image_name} | {safe_label} | "
            f"{probability:.4f} | {status}\n"
        )


def main():
    st.title("🌿 Plant Doctor AI")
    st.write(
        "Upload a plant leaf photo to explore a possible disease classification "
        "and general plant-care guidance."
    )

    try:
        model, idx_to_class, transform, device, model_version = load_artifacts()
    except (FileNotFoundError, RuntimeError, KeyError) as error:
        st.error(f"Could not load the model: {error}")
        st.stop()

    st.caption(f"Device: {device} · Model: {model_version}")

    uploaded_file = st.file_uploader(
        "Upload a clear leaf image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info("For better results, photograph one leaf clearly in good lighting.")
        return

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("This image could not be opened. Please upload a valid JPG or PNG.")
        return

    st.subheader("Uploaded Leaf")
    st.image(image, use_container_width=True)

    quality_warnings = image_quality_checks(image)

    if quality_warnings:
        st.warning("Image quality check")
        for warning in quality_warnings:
            st.write(f"• {warning}")
    else:
        st.caption("No obvious image-quality issue detected by the basic checks.")

    if st.button("Analyze Leaf", type="primary", use_container_width=True):
        
        with st.spinner("Checking image and analyzing the leaf..."):
            try:
                best_description, best_score, plant_score = validate_plant_image(image)
            except Exception as error:
                st.error(f"Image validation failed: {error}")
                st.info("Please try again. Disease prediction was not run.")
                return

            st.caption(
                f"Image validator: {best_description} "
                f"(relative score: {best_score:.1%})"
            )

            plant_descriptions = {
                "a close-up photograph of a plant leaf",
                "a photograph of a plant with visible leaves",
            }

            if best_description not in plant_descriptions:
                st.error(
                    "This image may not be a plant or leaf photo. "
                    "Please upload a clear image focused on one plant leaf."
                )
                st.info(
                    "The disease classifier was not run because the image "
                    "validator did not select a plant description."
                )
                return

            image_tensor = transform(image).unsqueeze(0).to(device)

            predictions = predict_topk(
                model=model,
                image_tensor=image_tensor,
                idx_to_class=idx_to_class,
                device=device,
                top_k=5,
            )


        if not predictions:
            st.error("The model did not return a prediction.")
            return

        top_label, top_prob = predictions[0]
        plant, disease, display_name = readable_name(top_label)

        low_confidence = top_prob < LOW_CONFIDENCE_THRESHOLD
        status = "LOW_CONFIDENCE" if low_confidence else "MODEL_PREDICTION"

        log_prediction(
            model_version,
            uploaded_file.name,
            top_label,
            top_prob,
            status,
        )

        st.divider()
        st.subheader("Analysis Result")

        if low_confidence:
            st.warning("The model is uncertain about this image.")
        else:
            st.success("The model produced a relatively high-confidence prediction.")

        st.markdown(f"### {display_name}")
        st.write(f"**Plant:** {plant}")
        st.write(f"**Predicted condition:** {disease}")
        st.metric("Model confidence score", f"{top_prob:.2%}")

        st.caption(
            "This score is derived from the model's output. It is not a verified "
            "probability that the diagnosis is correct. The model can be wrong "
            "even when its score is high."
        )

        if quality_warnings:
            st.warning(
                "Image-quality concerns may affect this result. Consider "
                "uploading a clearer image before relying on the prediction."
            )

        st.divider()
        st.subheader("Plant Care Guidance")

        guidance = disease_guidance(top_label)

        st.write(f"**Classification category:** {guidance['status']}")
        st.write(f"**About this result:** {guidance['summary']}")
        st.write(f"**Symptoms to check:** {guidance['symptoms']}")
        st.write(f"**Suggested next step:** {guidance['action']}")
        st.write(f"**Prevention:** {guidance['prevention']}")

        st.info(
            "This guidance is general educational information. The model has not "
            "confirmed the cause of the symptoms. Do not apply pesticides or "
            "discard a plant based only on this prediction."
        )

        st.divider()
        st.subheader("Top 5 Predictions")

        for label, probability in predictions:
            _, _, name = readable_name(label)
            st.write(f"**{name}** — {probability:.2%}")
            st.progress(max(0.0, min(1.0, float(probability))))

        st.caption(
            "Reliability limitation: this model classifies images into its "
            "known classes. These checks do not reliably detect every unsupported "
            "plant, unrelated image, or incorrect high-confidence prediction."
        )


if __name__ == "__main__":
    main()
