import base64
import io
from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image

from extract_features import extract_custom_features

MODEL_PATH = "model_rf.joblib"
SCALER_PATH = "scaler.joblib"
CLASS_NAMES = {0: "Real Art", 1: "AI Art"}
REACTION_IMAGES = {0: "asset/smiling.png", 1: "asset/crying.png"}
REACTION_SOUNDS = {0: "asset/goodmeme.mp3", 1: "asset/fartmeme.mp3"}


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def play_hidden_audio(path: str) -> None:
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode()
    st.markdown(
        f'<audio autoplay src="data:audio/mp3;base64,{b64}"></audio>',
        unsafe_allow_html=True,
    )


def preprocess(pil_image: Image.Image) -> np.ndarray:
    image_rgb = np.array(pil_image.convert("RGB"))
    image_rgb = cv2.resize(image_rgb, (256, 256))
    return image_rgb


def predict(model, scaler, image_rgb: np.ndarray):
    features = extract_custom_features(image_rgb).reshape(1, -1)
    features_scaled = scaler.transform(features)
    pred = int(model.predict(features_scaled)[0])

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features_scaled)[0]
    return pred, proba


def main():
    st.set_page_config(page_title="Real vs AI Art Classifier", page_icon=":art:")
    st.title("Real vs AI Art Classifier")
    st.write(
        "Upload an image and the model will predict whether it is real artwork "
        "or AI-generated"
    )

    try:
        model, scaler = load_artifacts()
    except FileNotFoundError:
        st.error(
            f"Could not find `{MODEL_PATH}` or `{SCALER_PATH}`. "
            "Run the training notebook first to produce them."
        )
        return

    uploaded = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )
    if uploaded is None:
        return

    image_bytes = uploaded.read()
    pil_image = Image.open(io.BytesIO(image_bytes))
    st.image(pil_image, caption="Input image", use_container_width=True)

    with st.spinner("Extracting features and predicting..."):
        image_rgb = preprocess(pil_image)
        pred, proba = predict(model, scaler, image_rgb)

    label = CLASS_NAMES[pred]
    st.subheader(f"Prediction: {label}")
    st.image(REACTION_IMAGES[pred], caption=label, width=200)
    play_hidden_audio(REACTION_SOUNDS[pred])

    if proba is not None:
        st.write("Class probabilities:")
        st.bar_chart(
            {CLASS_NAMES[i]: float(p) for i, p in enumerate(proba)},
        )


if __name__ == "__main__":
    main()
