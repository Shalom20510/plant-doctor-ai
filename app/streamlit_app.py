
import os
import sys
from pathlib import Path
from datetime import datetime

import streamlit as st
import torch
from PIL import Image, ImageFilter, ImageStat, ImageOps
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
    layout="wide",
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

    # Unusual framing can mean the leaf is too small in the frame. Rotation
    # itself is not proof of poor quality; EXIF orientation is normalized below.
    aspect_ratio = max(width, height) / max(1, min(width, height))
    if aspect_ratio > 2.4:
        warnings.append(
            "The framing is unusually wide or tall. Reframe one leaf so it fills "
            "more of the image if possible."
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

    # Approximate clutter warning: compare edge activity near the image border
    # with the center. This is a heuristic, not a validated segmentation model.
    border = max(1, min(width, height) // 10)
    if width > 2 * border and height > 2 * border:
        border_mask = Image.new("L", gray.size, 0)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(border_mask)
        draw.rectangle((0, 0, width - 1, border), fill=255)
        draw.rectangle((0, height - border, width - 1, height - 1), fill=255)
        draw.rectangle((0, 0, border, height - 1), fill=255)
        draw.rectangle((width - border, 0, width - 1, height - 1), fill=255)
        edge_gray = gray.filter(ImageFilter.FIND_EDGES)
        border_pixels = [
            pixel for pixel, mask in zip(edge_gray.getdata(), border_mask.getdata()) if mask
        ]
        center = gray.crop((width // 4, height // 4, 3 * width // 4, 3 * height // 4))
        center_edges = center.filter(ImageFilter.FIND_EDGES)
        border_activity = sum(border_pixels) / max(1, len(border_pixels))
        center_activity = ImageStat.Stat(center_edges).mean[0]
        if border_activity > max(45, center_activity * 1.8):
            warnings.append(
                "The background may be visually busy. Move closer to one leaf and "
                "use a simpler background if possible."
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


def inject_human_ui():
    """Apply a clean, accessible botanical theme without changing model behavior."""
    st.markdown(
        """
        <style>
        :root {
            --leaf: #426747;
            --leaf-dark: #203d2b;
            --ink: #26332a;
            --muted: #647064;
            --paper: #f4f5ed;
            --surface: #fffefa;
            --line: #dce4d5;
            --sage: #e7eddf;
        }
        .stApp {
            background: var(--paper);
            color: var(--ink);
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        [data-testid="stHeader"] {
            background: rgba(244,245,237,.96);
        }
        .block-container {
            max-width: 1180px;
            padding-top: 1.35rem;
            padding-bottom: 3.5rem;
        }
        h1, h2, h3, h4 {
            color: var(--leaf-dark);
            letter-spacing: -.025em;
        }
        h1, h2 { line-height: 1.15; }
        h2 { font-size: clamp(1.8rem, 3vw, 2.35rem) !important; }
        h3, h4 { letter-spacing: -.015em; }
        p, li, label, [data-testid="stMarkdownContainer"] {
            color: var(--ink);
        }
        .plant-brand {
            display: flex;
            align-items: baseline;
            gap: .6rem;
            flex-wrap: wrap;
            color: var(--leaf-dark);
            font-size: 1.35rem;
            font-weight: 750;
            letter-spacing: -.025em;
            padding: .35rem 0 .8rem;
        }
        .plant-brand span {
            color: var(--muted);
            font-size: .98rem;
            font-weight: 400;
            letter-spacing: 0;
        }
        .plant-eyebrow {
            text-transform: uppercase;
            letter-spacing: .15em;
            font-size: .72rem;
            font-weight: 750;
            color: var(--leaf);
            margin: .2rem 0 1rem;
        }
        .plant-intro {
            max-width: 600px;
            color: var(--muted);
            font-size: 1.06rem;
            line-height: 1.75;
        }
        .plant-note {
            background: var(--sage);
            border-left: 3px solid #78916d;
            padding: 1rem 1.1rem;
            color: #344535;
            border-radius: 0 10px 10px 0;
            font-size: .94rem;
            line-height: 1.6;
            margin: 1.1rem 0;
        }
        .example-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1.1rem;
            overflow: hidden;
        }
        .example-art {
            background: #e9efe2;
            border-radius: 13px;
            padding: .5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 290px;
        }
        .example-title {
            color: var(--leaf-dark);
            font-size: 1.12rem;
            font-weight: 700;
            margin: .9rem 0 .25rem;
        }
        .example-copy {
            color: var(--muted);
            font-size: .9rem;
            line-height: 1.55;
            margin: 0;
        }
        /* Streamlit uploader: force a light dropzone and readable dark text. */
        [data-testid="stFileUploader"] {
            background: var(--surface) !important;
            border: 1px dashed #a9b8a1 !important;
            border-radius: 14px !important;
            padding: 1rem !important;
        }
        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploaderDropzone"],
        [data-testid="stFileUploaderDropzone"] > div {
            background: var(--surface) !important;
            border: 0 !important;
        }
        [data-testid="stFileUploader"] *,
        [data-testid="stFileUploaderDropzoneInstructions"] *,
        [data-testid="stFileUploader"] small {
            color: var(--ink) !important;
            opacity: 1 !important;
        }
        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploader"] button p,
        [data-testid="stFileUploader"] button span {
            background: var(--leaf) !important;
            color: #ffffff !important;
            border: 1px solid var(--leaf) !important;
            opacity: 1 !important;
        }
        [data-testid="stFileUploader"] button {
            border-radius: 8px !important;
        }
        [data-testid="stFileUploader"] button:hover {
            background: var(--leaf-dark) !important;
            border-color: var(--leaf-dark) !important;
        }
        .stButton > button[kind="primary"] {
            background: var(--leaf) !important;
            color: #ffffff !important;
            border: 1px solid var(--leaf) !important;
            border-radius: 9px !important;
            min-height: 2.8rem;
            font-weight: 650;
        }
        .stButton > button[kind="primary"]:hover {
            background: var(--leaf-dark) !important;
            border-color: var(--leaf-dark) !important;
        }
        .stButton > button { border-radius: 9px; }
        [data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--line);
            padding: 1rem;
            border-radius: 12px;
        }
        [data-testid="stAlert"] { border-radius: 10px; }
        hr { border-color: var(--line); }
        footer { visibility: hidden; }
        @media (max-width: 700px) {
            .block-container { padding: 1rem 1rem 2.5rem; }
            .plant-brand { font-size: 1.18rem; }
            .plant-brand span { font-size: .88rem; }
            .example-art { min-height: 220px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def main():
    inject_human_ui()
    st.markdown('<div class="plant-brand">🌿 Plant Doctor <span style="font-weight:400;color:#687267">/ a little help for healthier plants</span></div>', unsafe_allow_html=True)
    st.divider()

    try:
        model, idx_to_class, transform, device, model_version = load_artifacts()
    except (FileNotFoundError, RuntimeError, KeyError) as error:
        st.error(f"Could not load the model: {error}")
        st.stop()

    st.markdown('<div class="plant-eyebrow">A simple plant health check</div>', unsafe_allow_html=True)
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.markdown("## What's happening to your plant?")
        st.markdown(
            '<p class="plant-intro">Noticed something unusual on a leaf? Add a photo and we’ll help you explore what might be happening, with practical next steps.</p>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="plant-note">For the clearest result, photograph one leaf in natural light and get close enough to see the details.</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a leaf photo",
            type=["jpg", "jpeg", "png"],
            help="JPG or PNG. A close, focused photo works best.",
            label_visibility="visible",
        )
        st.caption("JPG or PNG · Clear, well-lit leaf photos work best")
    with right:
        st.markdown(
            """
            <div class="example-card">
              <div class="example-art">
                <svg viewBox="0 0 420 320" role="img" aria-label="Illustration of a healthy green plant" style="width:100%;max-width:390px;height:auto;">
                  <ellipse cx="210" cy="282" rx="100" ry="12" fill="#d3dfc9"/>
                  <path d="M210 253 C207 211 212 171 207 112" stroke="#426747" stroke-width="7" stroke-linecap="round" fill="none"/>
                  <path d="M207 181 C168 171 147 144 145 111 C177 116 204 139 207 181Z" fill="#71956a"/>
                  <path d="M208 149 C237 140 263 111 263 79 C231 88 211 111 208 149Z" fill="#426747"/>
                  <path d="M209 218 C173 213 148 191 143 164 C176 166 202 187 209 218Z" fill="#9bb68b"/>
                  <path d="M210 205 C246 197 271 173 277 145 C244 151 218 175 210 205Z" fill="#62875b"/>
                  <path d="M208 124 C188 111 180 88 187 65 C208 79 218 101 208 124Z" fill="#b3c9a4"/>
                  <path d="M210 253 C210 236 211 224 210 211" stroke="#426747" stroke-width="5" stroke-linecap="round" fill="none"/>
                  <path d="M157 132 L202 171 M254 96 L213 139 M157 181 L202 210 M266 158 L216 197 M190 76 L207 117" stroke="#d9e6d0" stroke-width="2.5" stroke-linecap="round" fill="none"/>
                  <path d="M153 246 L267 246 L253 295 Q210 309 167 295 Z" fill="#c88461"/>
                  <path d="M146 239 Q210 230 274 239 L270 254 Q210 264 150 254 Z" fill="#a96547"/>
                  <path d="M177 269 Q210 277 243 269" stroke="#dba184" stroke-width="3" stroke-linecap="round" fill="none"/>
                  <circle cx="307" cy="61" r="20" fill="#d3dfc9"/>
                  <path d="M300 61 Q307 48 314 61 Q307 74 300 61Z" fill="#426747"/>
                </svg>
              </div>
              <div class="example-title">What a good photo should show</div>
              <p class="example-copy">One leaf in focus, clear details, and soft natural daylight. This illustration is a guide to framing, not a real plant photo.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


    st.divider()
    if uploaded_file is None:
        st.markdown("#### A few tips before you take the photo")
        tip1, tip2, tip3 = st.columns(3)
        tip1.markdown("**Get close**  \nKeep one leaf large in the frame.")
        tip2.markdown("**Use daylight**  \nAvoid strong shadows and glare.")
        tip3.markdown("**Keep it steady**  \nKeep the leaf details sharp.")
        st.caption("Your photo is checked for basic quality before the disease classifier runs.")
        return

    try:
        source_image = Image.open(uploaded_file)
        exif_orientation = source_image.getexif().get(274, 1)
        image = ImageOps.exif_transpose(source_image).convert("RGB")
    except Exception:
        st.error("This image could not be opened. Please upload a valid JPG or PNG.")
        return

    st.subheader("Your photo")
    photo_col, space_col = st.columns([0.72, 0.28], gap="large")
    with photo_col:
        st.image(image, width="stretch")
    with space_col:
        st.markdown("**Before analysis**")
        st.caption("Check that the leaf is visible and the affected area is in focus.")
        st.caption("The image-quality checks below may recommend a clearer photo.")

    quality_warnings = image_quality_checks(image)
    if exif_orientation in (3, 6, 8):
        quality_warnings.append(
            "The photo contained rotated orientation metadata; the app corrected "
            "the display orientation automatically."
        )

    if quality_warnings:
        st.warning("Image quality check")
        for warning in quality_warnings:
            st.write(f"• {warning}")
    else:
        st.caption("No obvious image-quality issue detected by the basic checks.")

    if st.button("Analyze Leaf", type="primary", width="stretch"):
        
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

            # This relative CLIP score is a conservative screening heuristic,
            # not a calibrated probability of plant identity.
            if plant_score < 0.30:
                st.error(
                    "The image is too ambiguous for a responsible prediction. "
                    "Please upload a clearer, well-lit photo focused on one leaf."
                )
                st.info("Disease prediction was not run because plant-image confidence was low.")
                return

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

            severe_quality_terms = (
                "low resolution",
                "very dark",
                "overexposed",
                "blurry or lack detail",
            )
            if any(
                any(term in warning.lower() for term in severe_quality_terms)
                for warning in quality_warnings
            ):
                st.error(
                    "This image may be too degraded for a reliable prediction. "
                    "Please upload a sharper, properly lit image with the leaf in focus."
                )
                st.info("Disease prediction was not run. Improve the image and try again.")
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
        st.markdown("## What we noticed")
        result_image, result_text = st.columns([0.9, 1.1], gap="large")
        with result_image:
            st.image(image, width="stretch")
            st.caption("Your uploaded leaf")
        with result_text:
            if low_confidence:
                st.warning("The model is unsure about this photo. Treat this as a starting point, not a diagnosis.")
            else:
                st.info("Here’s the model’s closest match. It may still be wrong, so compare it with the symptoms you can see.")
            st.markdown(f"### {display_name}")
            st.write(f"**Plant:** {plant}")
            st.write(f"**Possible condition:** {disease}")
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
        st.subheader("Other possible matches")
        st.caption("These are the model's next closest classes, not confirmed diagnoses.")
        for rank, (label, probability) in enumerate(predictions, start=1):
            _, _, name = readable_name(label)
            if rank == 1:
                continue
            row_name, row_score = st.columns([4, 1])
            row_name.write(name)
            row_score.write(f"{probability:.1%}")

        st.caption(
            "Reliability limitation: this model classifies images into its "
            "known classes. These checks do not reliably detect every unsupported "
            "plant, unrelated image, or incorrect high-confidence prediction."
        )


if __name__ == "__main__":
    main()
