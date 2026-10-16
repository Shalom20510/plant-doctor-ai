"""Streamlit entry point for Plant Doctor AI.

The interface and inference workflow will be implemented in a later step.
"""

import streamlit as st


def main() -> None:
    st.set_page_config(page_title="Plant Doctor AI", page_icon="🌿", layout="centered")
    st.title("🌿 Plant Doctor AI")
    st.write("Project foundation is ready. Image upload and prediction will be added next.")


if __name__ == "__main__":
    main()
