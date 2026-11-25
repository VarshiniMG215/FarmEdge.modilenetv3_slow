import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image, ImageOps

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
CLASS_FILE = MODEL_DIR / "class_names.json"
MODEL_OPTIONS = {
    "MobileNetV3-Small (FP32 — recommended)": MODEL_DIR / "mobilenetv3_small_fp32.onnx",
    "MobileNetV3-Small (FP16 — smaller edge model)": MODEL_DIR / "mobilenetv3_small_edge_final.onnx",
}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

st.set_page_config(page_title="FarmEdge | Plant Disease Detection", page_icon="🌿", layout="wide")
st.title("🌿 FarmEdge")
st.subheader("Lightweight Plant Disease Detection")
st.write("Upload a clear image of a plant leaf to predict its likely disease class using the trained MobileNetV3-Small model.")
st.info("This is an academic prototype. Predictions may be wrong for real-world photos, unusual lighting, backgrounds, or diseases not represented in the training dataset. Do not use it as the sole basis for crop-treatment decisions.")

if not CLASS_FILE.exists():
    st.error(f"Class labels file not found: {CLASS_FILE}")
    st.stop()
with open(CLASS_FILE, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)

@st.cache_resource(show_spinner="Loading model…")
def load_session(model_path_str: str):
    model_path = Path(model_path_str)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}. Keep the models folder beside app.py.")
    # FP32 is most compatible. FP16 is offered for edge-size demonstration.
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

def preprocess(image: Image.Image) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    arr = (arr - MEAN) / STD
    return np.ascontiguousarray(arr, dtype=np.float32)

def softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64).reshape(-1)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)

with st.sidebar:
    st.header("Settings")
    model_label = st.selectbox("Inference model", list(MODEL_OPTIONS.keys()), index=0)
    st.caption("FP32 is recommended if the FP16 model fails to load in your ONNX Runtime installation.")
    st.divider()
    st.write("**Model input:** RGB image resized to 224 × 224")
    st.write("**Classes:** 38 plant/disease categories")
    st.write("**Runtime:** ONNX Runtime on CPU")

uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png", "webp", "bmp"])
if uploaded is not None:
    try:
        image = Image.open(uploaded)
        left, right = st.columns([1, 1])
        with left:
            st.image(image, caption="Uploaded image", use_container_width=True)
        with st.spinner("Analyzing image…"):
            session = load_session(str(MODEL_OPTIONS[model_label]))
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            x = preprocess(image)
            start = time.perf_counter()
            output = session.run([output_name], {input_name: x})[0]
            elapsed_ms = (time.perf_counter() - start) * 1000
            probs = softmax(output)
            if len(probs) != len(CLASS_NAMES):
                raise ValueError(f"Model returned {len(probs)} scores, but class_names.json contains {len(CLASS_NAMES)} labels.")
            top_indices = np.argsort(probs)[::-1][:3]
        with right:
            predicted = CLASS_NAMES[int(top_indices[0])]
            plant, _, disease = predicted.partition("___")
            st.success("Prediction completed")
            st.markdown("### Predicted class")
            st.write(f"**Plant:** {plant.replace('_', ' ')}")
            st.write(f"**Class:** {disease.replace('_', ' ').strip() if disease else predicted.replace('_', ' ')}")
            st.metric("Model score", f"{probs[top_indices[0]] * 100:.2f}%")
            st.metric("Inference time", f"{elapsed_ms:.2f} ms/image")
            st.caption("The model score is not a calibrated probability and should not be interpreted as guaranteed confidence.")
            st.markdown("### Top 3 predictions")
            top_rows = []
            for idx in top_indices:
                name = CLASS_NAMES[int(idx)].replace("___", " — ").replace("_", " ").strip()
                top_rows.append({"Class": name, "Score (%)": round(float(probs[idx] * 100), 2)})
            st.dataframe(top_rows, hide_index=True, use_container_width=True)
    except Exception as e:
        st.error(f"Could not process this image: {e}")
        st.warning("Try the FP32 model in the sidebar. Also check that the models folder and class_names.json are present.")
else:
    st.markdown("#### How to use")
    st.markdown("1. Upload a JPG, PNG, WEBP, or BMP image.\n2. Keep the recommended FP32 model selected initially.\n3. Review the predicted class and top three model scores.")

st.divider()
st.caption("FarmEdge student project • MobileNetV3-Small • New Plant Diseases dataset (38 classes)")
