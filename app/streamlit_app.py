
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

from src.input_validation import (
    CROP_PROMPT_VARIANTS, NON_LEAF_PROMPTS, SUPPORTED_CROP_NAMES,
    SUPPORTED_PROMPTS, UNSUPPORTED_LEAF_PROMPTS, aggregate_scope_logits,
    decide_input_scope, crop_name_from_prompt, normalize_crop_name,
)

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

LANGUAGES = {"English": "en", "తెలుగు": "te"}
TRANSLATIONS = {
    "en": {},
    "te": {
        "Language / భాష": "Language / భాష",
        "A simple plant health check": "మొక్క ఆరోగ్యాన్ని సులభంగా పరీక్షించండి",
        "What's happening to your plant?": "మీ మొక్కకు ఏమైంది?",
        tr("Choose a leaf photo"): "ఆకు ఫోటోను ఎంచుకోండి",
        "JPG or PNG · Clear, well-lit leaf photos work best": "JPG లేదా PNG · స్పష్టమైన, మంచి వెలుతురు ఉన్న ఫోటోలు ఉత్తమం",
        "A few tips before you take the photo": "ఫోటో తీయడానికి ముందు కొన్ని సూచనలు",
        "Your photo": "మీ ఫోటో",
        "Before analysis": "విశ్లేషణకు ముందు",
        "Image quality check": "చిత్ర నాణ్యత తనిఖీ",
        "Analyze Leaf": "ఆకును విశ్లేషించండి",
        "Inconclusive result": "నిర్ధారణకు సరిపడ సమాచారం లేదు",
        "Your uploaded leaf": "మీరు అప్‌లోడ్ చేసిన ఆకు",
        "What we noticed": "మేము గమనించినది",
        "Model confidence score": "మోడల్ విశ్వాస స్కోరు",
        "Plant Care Guidance": "మొక్క సంరక్షణ సూచనలు",
        "Other possible matches": "ఇతర సంభావ్య సరిపోలికలు",
        "Closest model categories (not diagnoses):": "మోడల్‌కు దగ్గరగా ఉన్న వర్గాలు (నిర్ధారణలు కావు):",
        "These are the model's next closest classes, not confirmed diagnoses.": "ఇవి మోడల్‌కు దగ్గరగా ఉన్న ఇతర వర్గాలు మాత్రమే; నిర్ధారిత వ్యాధి నిర్ధారణలు కావు.",
        "Please try again. Disease prediction was not run.": "మళ్లీ ప్రయత్నించండి. వ్యాధి అంచనా చేయలేదు.",
        "Plant": "మొక్క",
        "Possible condition": "సంభావ్య సమస్య",
        "Classification category": "వర్గీకరణ",
        "About this result": "ఈ ఫలితం గురించి",
        "Symptoms to check": "పరిశీలించాల్సిన లక్షణాలు",
        "Suggested next step": "తదుపరి సూచన",
        "Prevention": "నివారణ",
    },
}
def tr(text):
    """Translate selected UI labels; model predictions remain unchanged."""
    language = st.session_state.get("plant_doctor_language", "en")
    return TRANSLATIONS.get(language, {}).get(text, text)


@st.cache_resource
def load_clip_validator():
    model_name = "openai/clip-vit-base-patch32"

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()

    return processor, model


def validate_plant_image(image):
    """Screen inputs using averaged multi-template CLIP scores per crop."""
    processor, model = load_clip_validator()

    crop_prompt_pairs = [
        (crop, prompt)
        for crop in SUPPORTED_CROP_NAMES
        for prompt in CROP_PROMPT_VARIANTS[crop]
    ]
    crop_descriptions = [prompt for _, prompt in crop_prompt_pairs]
    descriptions = crop_descriptions + list(UNSUPPORTED_LEAF_PROMPTS) + list(NON_LEAF_PROMPTS)
    inputs = processor(text=descriptions, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image[0]

    # Keep the raw prompt logits and let the tested helper average within each
    # crop, then compare supported crops against unsupported leaf/non-leaf groups.
    crop_prompt_scores = {}
    offset = 0
    for crop in SUPPORTED_CROP_NAMES:
        prompt_count = len(CROP_PROMPT_VARIANTS[crop])
        crop_prompt_scores[crop] = [
            float(value.item()) for value in logits[offset:offset + prompt_count]
        ]
        offset += prompt_count

    unsupported_start = len(crop_descriptions)
    unsupported_end = unsupported_start + len(UNSUPPORTED_LEAF_PROMPTS)
    unsupported_prompt_scores = [
        float(value.item()) for value in logits[unsupported_start:unsupported_end]
    ]
    nonleaf_start = unsupported_end
    non_leaf_prompt_scores = {
        prompt: float(logits[nonleaf_start + index].item())
        for index, prompt in enumerate(NON_LEAF_PROMPTS)
    }

    return aggregate_scope_logits(
        crop_prompt_scores,
        unsupported_prompt_scores,
        non_leaf_prompt_scores,
    )



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
            padding: 1rem;
            overflow: hidden;
        }
        .guide-visual {
            background: #e9efe2;
            border-radius: 13px;
            overflow: hidden;
            padding: .4rem;
        }
        .guide-heading {
            color: var(--leaf-dark);
            font-size: 1.12rem;
            font-weight: 750;
            margin: .85rem 0 .2rem;
        }
        .guide-subtitle {
            color: var(--muted);
            font-size: .9rem;
            line-height: 1.5;
            margin-bottom: .85rem;
        }
        .guide-examples {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: .55rem;
        }
        .guide-example {
            min-width: 0;
            border-radius: 10px;
            padding: .6rem .65rem;
            background: #f0f5eb;
            border: 1px solid #e0e9d9;
        }
        .guide-example.bad {
            background: #fff1ef;
            border-color: #f4dfdc;
        }
        .guide-example strong {
            display: block;
            color: #315b3b;
            font-size: .78rem;
            margin-bottom: .18rem;
        }
        .guide-example.bad strong { color: #a33f3a; }
        .guide-example span {
            display: block;
            color: #586458;
            font-size: .73rem;
            line-height: 1.4;
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
    language_label = st.selectbox("Language / భాష", options=list(LANGUAGES), key="plant_doctor_language_label")
    st.session_state["plant_doctor_language"] = LANGUAGES[language_label]
    inject_human_ui()
    st.markdown('<div class="plant-brand">🌿 Plant Doctor <span style="font-weight:400;color:#687267">/ a little help for healthier plants</span></div>', unsafe_allow_html=True)
    st.divider()

    try:
        model, idx_to_class, transform, device, model_version = load_artifacts()
    except (FileNotFoundError, RuntimeError, KeyError) as error:
        st.error(f"Could not load the model: {error}")
        st.stop()

    st.markdown(f'<div class="plant-eyebrow">{tr("A simple plant health check")}</div>', unsafe_allow_html=True)
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.markdown(f"## {tr(\"What's happening to your plant?\")}")
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
        st.caption(tr("JPG or PNG · Clear, well-lit leaf photos work best"))
    with right:
        st.markdown(
            """
            <div class="example-card">
              <div class="guide-visual">
                <svg viewBox="0 0 520 300" role="img" aria-label="Leaf photo framing guide showing one leaf in focus, daylight and a clear close-up" style="display:block;width:100%;height:auto;">
                  <defs>
                    <linearGradient id="leafBg" x1="0" y1="0" x2="1" y2="1">
                      <stop offset="0" stop-color="#dfe9d5"/>
                      <stop offset="1" stop-color="#cbdcbd"/>
                    </linearGradient>
                    <linearGradient id="leafFill" x1="0" y1="0" x2="1" y2="1">
                      <stop offset="0" stop-color="#9fbd79"/>
                      <stop offset="1" stop-color="#3d7447"/>
                    </linearGradient>
                  </defs>
                  <rect width="520" height="300" rx="12" fill="url(#leafBg)"/>
                  <circle cx="444" cy="53" r="25" fill="#f8f7e9" opacity=".9"/>
                  <circle cx="444" cy="53" r="11" fill="#e8bb56"/>
                  <g stroke="#e8bb56" stroke-width="3" stroke-linecap="round">
                    <path d="M444 17V10 M444 96V89 M408 53H401 M487 53H480 M419 28L414 23 M469 78L474 83 M469 28L474 23 M419 78L414 83"/>
                  </g>
                  <path d="M251 270 C249 218 254 172 276 117" stroke="#426747" stroke-width="8" stroke-linecap="round" fill="none"/>
                  <path d="M267 190 C203 188 159 151 151 91 C210 93 259 130 267 190Z" fill="url(#leafFill)"/>
                  <path d="M269 166 C315 148 350 105 349 56 C302 71 273 111 269 166Z" fill="#426747"/>
                  <path d="M259 234 C208 231 173 205 161 164 C210 164 247 193 259 234Z" fill="#79a164"/>
                  <path d="M269 220 C315 206 346 176 355 137 C310 143 280 179 269 220Z" fill="#5e8d50"/>
                  <path d="M266 190 L169 104 M270 165 L337 68 M258 231 L176 177 M270 219 L343 147" stroke="#d8e9c9" stroke-width="3" stroke-linecap="round" fill="none"/>
                  <rect x="118" y="32" width="284" height="232" rx="10" fill="none" stroke="#fffefa" stroke-width="3" stroke-dasharray="8 7" opacity=".95"/>
                  <path d="M118 55V32H141 M379 32H402V55 M118 241V264H141 M379 264H402V241" fill="none" stroke="#315b3b" stroke-width="5" stroke-linecap="round"/>
                  <rect x="18" y="18" width="116" height="38" rx="19" fill="#fffefa" opacity=".96"/>
                  <circle cx="38" cy="37" r="8" fill="#426747"/>
                  <path d="M34 37L37 40L43 33" fill="none" stroke="#fffefa" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                  <text x="53" y="42" font-size="12" font-family="Arial, sans-serif" font-weight="700" fill="#315b3b">In focus</text>
                  <rect x="18" y="246" width="146" height="36" rx="18" fill="#fffefa" opacity=".96"/>
                  <circle cx="38" cy="264" r="8" fill="#e8bb56"/>
                  <text x="53" y="269" font-size="12" font-family="Arial, sans-serif" font-weight="700" fill="#315b3b">Natural daylight</text>
                  <rect x="344" y="235" width="158" height="42" rx="10" fill="#fffefa" opacity=".96"/>
                  <text x="358" y="252" font-size="11" font-family="Arial, sans-serif" font-weight="700" fill="#315b3b">One leaf, close-up</text>
                  <text x="358" y="266" font-size="10" font-family="Arial, sans-serif" fill="#586458">Show spots and edges</text>
                </svg>
              </div>
              <div class="guide-heading">How to take a useful leaf photo</div>
              <div class="guide-subtitle">Fill the frame with one leaf so the affected area is easy to see.</div>
              <div class="guide-examples">
                <div class="guide-example">
                  <strong>✓ Good photo</strong>
                  <span>One clear leaf, sharp details, natural light.</span>
                </div>
                <div class="guide-example bad">
                  <strong>✕ Too blurry</strong>
                  <span>Hold the camera steady and tap to focus.</span>
                </div>
                <div class="guide-example bad">
                  <strong>✕ Too dark</strong>
                  <span>Move to bright shade or soft daylight.</span>
                </div>
                <div class="guide-example bad">
                  <strong>✕ Too many leaves</strong>
                  <span>Frame just one leaf and its affected area.</span>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    if uploaded_file is None:
        st.markdown(f"#### {tr(\"A few tips before you take the photo\")}")
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

    st.subheader(tr("Your photo"))
    photo_col, space_col = st.columns([0.72, 0.28], gap="large")
    with photo_col:
        st.image(image, width="stretch")
    with space_col:
        st.markdown(f"**{tr(\"Before analysis\")}**")
        st.caption("Check that the leaf is visible and the affected area is in focus.")
        st.caption("The image-quality checks below may recommend a clearer photo.")

    quality_warnings = image_quality_checks(image)
    if exif_orientation in (3, 6, 8):
        quality_warnings.append(
            "The photo contained rotated orientation metadata; the app corrected "
            "the display orientation automatically."
        )

    if quality_warnings:
        st.warning(tr("Image quality check"))
        for warning in quality_warnings:
            st.write(f"• {warning}")
    else:
        st.caption("No obvious image-quality issue detected by the basic checks.")

    st.caption(
        "Plant identification is automatic. If the crop screen and disease model disagree, "
        "the app will return an inconclusive result instead of a disease prediction."
    )

    if st.button(tr("Analyze Leaf"), type="primary", width="stretch"):
        
        with st.spinner("Checking image and analyzing the leaf..."):
            try:
                best_description, best_score, supported_score = validate_plant_image(image)
            except Exception as error:
                st.error(f"Image validation failed: {error}")
                st.info(tr("Please try again. Disease prediction was not run."))
                return

            scope = decide_input_scope(best_description, supported_score, best_score)
            st.caption(
                f"Input scope check: {scope['status'].replace('_', ' ').title()} "
                f"(winning scope-category score: {best_score:.1%}; supported-crop scope score: {supported_score:.1%})."
            )
            st.caption(
                "These are relative CLIP scores, not calibrated probabilities. "
                "The scope check can accept or reject images incorrectly."
            )
            if not scope["accepted"]:
                st.warning(f"Inconclusive / unsupported input: {scope['reason']}")
                log_prediction(
                    model_version, uploaded_file.name, "NO_PREDICTION",
                    float(best_score), f"REJECTED_{str(scope['status']).upper()}",
                )
                st.info(
                    "Supported crops represented in this model: apple, blueberry, cherry, corn, "
                    "grape, orange, peach, bell pepper, potato, raspberry, soybean, squash, "
                    "strawberry, and tomato."
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

        # Cross-check the crop selected by CLIP against the crop predicted by
        # the disease classifier. This is a conservative heuristic, not proof
        # that either model identified the plant correctly.
        if best_description in SUPPORTED_PROMPTS:
            clip_crop = crop_name_from_prompt(best_description)
            if (
                clip_crop is not None
                and normalize_crop_name(clip_crop) != normalize_crop_name(plant)
            ):
                log_prediction(
                    model_version,
                    uploaded_file.name,
                    top_label,
                    top_prob,
                    "REJECTED_CROP_MISMATCH",
                )
                st.divider()
                st.markdown(f"## {tr(\"Inconclusive result\")}")
                mismatch_image, mismatch_text = st.columns([0.9, 1.1], gap="large")
                with mismatch_image:
                    st.image(image, width="stretch")
                    st.caption(tr("Your uploaded leaf"))
                with mismatch_text:
                    st.warning(
                        "Plant type mismatch: the image-screening model and disease "
                        "classifier disagree about the crop."
                    )
                    st.write(f"**Image-screening crop:** {clip_crop.title()}")
                    st.write(f"**Disease-model crop:** {plant}")
                    st.write(
                        "No disease diagnosis is being reported because the crop "
                        "predictions do not agree."
                    )
                    st.caption(
                        "This consistency check is a heuristic. Both models can be wrong, "
                        "and matching crop labels still do not guarantee a correct disease "
                        "prediction. Try a sharp, close-up photo of one supported crop leaf."
                    )
                return

        low_confidence = top_prob < LOW_CONFIDENCE_THRESHOLD
        if low_confidence:
            log_prediction(
                model_version, uploaded_file.name, top_label, top_prob,
                "REJECTED_LOW_CONFIDENCE",
            )
            st.divider()
            st.warning(
                f"Inconclusive result: the top model score is {top_prob:.1%}, below "
                f"the configured {LOW_CONFIDENCE_THRESHOLD:.0%} threshold. "
                "No disease diagnosis is being reported."
            )
            st.write(tr("Closest model categories (not diagnoses):"))
            for label, probability in predictions:
                _, _, name = readable_name(label)
                st.write(f"- {name}: {probability:.1%}")
            st.caption(
                "Softmax scores can be overconfident and are not calibrated probabilities. "
                "This threshold is a demo rejection rule, not a clinically or agriculturally "
                "validated guarantee. Try a clear, well-lit photo of one supported crop leaf."
            )
            return

        log_prediction(model_version, uploaded_file.name, top_label, top_prob, "MODEL_PREDICTION")

        st.divider()
        st.markdown(f"## {tr(\"What we noticed\")}")
        result_image, result_text = st.columns([0.9, 1.1], gap="large")
        with result_image:
            st.image(image, width="stretch")
            st.caption("Your uploaded leaf")
        with result_text:
            st.info(
                "Here’s the model’s closest match after the input checks. It may still be wrong, "
                "so compare it with visible symptoms."
            )
            st.markdown(f"### {display_name}")
            st.write(f"**Plant:** {plant}")
            st.write(f"**Possible condition:** {disease}")
            st.metric(tr("Model confidence score"), f"{top_prob:.2%}")

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
        st.subheader(tr("Plant Care Guidance"))

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
        st.subheader(tr("Other possible matches"))
        st.caption(tr("These are the model's next closest classes, not confirmed diagnoses."))
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
