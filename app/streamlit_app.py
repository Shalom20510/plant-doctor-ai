"""Streamlit interface for Plant Doctor AI."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.predict import predict_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "artifacts" / "model_v1.pt"


def main() -> None:
    st.set_page_config(
        page_title="Plant Doctor AI",
        page_icon="🌿",
        layout="centered",
    )

    st.title("🌿 Plant Doctor AI")
    st.write(
        "Upload a clear photo of a plant leaf to receive an AI-generated "
        "estimate of its likely health condition."
    )

    st.info(
        "This tool is a prototype. Predictions are only estimates and may be "
        "incorrect, especially for plant varieties or symptoms outside the "
        "model's training data. Confirm important treatment decisions with an "
        "agricultural expert."
    )

    uploaded_file = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png"],
        help="Use a clear, well-lit image with the leaf visible.",
    )

    checkpoint_path = st.text_input(
        "Model checkpoint path",
        value=str(DEFAULT_CHECKPOINT),
        help=(
            "Path to a compatible trained EfficientNet-B0 plant-disease "
            "checkpoint. The checkpoint is not included in this repository yet."
        ),
    )

    if uploaded_file is not None:
        st.subheader("Uploaded image")
        st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)

        if st.button("Analyze leaf", type="primary", use_container_width=True):
            checkpoint = Path(checkpoint_path).expanduser()
            if not checkpoint.is_absolute():
                checkpoint = PROJECT_ROOT / checkpoint

            if not checkpoint.is_file():
                st.error(
                    "A trained model checkpoint was not found at "
                    f"`{checkpoint}`. The prediction pipeline is connected, "
                    "but you need compatible trained weights before analysis "
                    "can run. We will handle model/data setup in a later step."
                )
                st.stop()

            with st.spinner("Analyzing the leaf image..."):
                try:
                    result = predict_image(
                        uploaded_file,
                        checkpoint_path=checkpoint,
                        top_k=3,
                    )
                except (FileNotFoundError, ValueError, TypeError, RuntimeError) as exc:
                    st.error(f"Could not analyze this image: {exc}")
                    st.stop()
                except Exception as exc:  # Keep unexpected inference errors visible in the UI.
                    st.error(
                        "An unexpected prediction error occurred. Check that the "
                        "checkpoint matches the expected model architecture."
                    )
                    st.caption(str(exc))
                    st.stop()

            st.subheader("Prediction")
            st.markdown(f"**Likely crop:** {result['crop']}")
            st.markdown(f"**Predicted condition:** {result['condition']}")
            st.metric("Model confidence", f"{result['confidence']:.1%}")

            if result["confidence"] < 0.60:
                st.warning(
                    "The model confidence is low. Do not rely on this result alone."
                )

            st.subheader("Other possible predictions")
            for index, item in enumerate(result["top_predictions"], start=1):
                st.write(
                    f"{index}. **{item['crop']} — {item['condition']}** "
                    f"({item['confidence']:.1%})"
                )

            st.caption(result["note"])


if __name__ == "__main__":
    main()
