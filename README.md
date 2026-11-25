# FarmEdge — Lightweight Plant Disease Detection

FarmEdge is a Streamlit prototype that classifies an uploaded plant-leaf image using a MobileNetV3-Small model exported to ONNX. It displays the predicted class, model score, inference time, and top-three classes.


# FarmEdge — Lightweight Plant Disease Detection

FarmEdge is a Streamlit-based web prototype for lightweight plant leaf disease classification using **MobileNetV3-Small** exported to **ONNX** and executed via **ONNX Runtime** on CPU. It is designed to demonstrate fast, low-resource inference without requiring heavy training frameworks or GPU hardware at runtime.

---

## Overview & Key Features(DDATAset is plantvillage dataset from kaggle)


* **38-Class Leaf Classification:** Identifies crop and health/disease categories across plant species such as Apple, Blueberry, Cherry, Corn (maize), Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.
* **Two ONNX Model Variants:** Switch between an FP32 baseline model (`mobilenetv3_small_fp32.onnx`) and an FP16 edge-optimized model (`mobilenetv3_small_edge_final.onnx`) directly from the Streamlit sidebar.
* **Deterministic Image Preprocessing:** Applies EXIF orientation correction, converts images to RGB, resizes to `224 × 224` using bilinear interpolation, scales pixels to `[0, 1]`, transposes tensors to `(1, 3, 224, 224)`, and normalizes with ImageNet mean (`[0.485, 0.456, 0.406]`) and standard deviation (`[0.229, 0.224, 0.225]`).
* **Real-Time Inference Feedback:** Displays the top-1 predicted crop and condition, inference latency in milliseconds (`ms`), a softmax score breakdown, and a top-3 predictions table.

---

## Dataset & Data Pipeline

* **Dataset:** PlantVillage leaf image dataset spanning **38** plant-and-condition classes mapped in `models/class_names.json` (formatted as `<Crop>___<Condition>`).
* **Data Split:** 80% training, 10% validation, and 10% test split.
* **Input Standardization:** All model inputs are resized to `224 × 224` RGB and normalized consistently during inference so results match expected input distributions.

---

## Model Architecture & Edge Optimization

* **Backbone:** **MobileNetV3-Small**, chosen for low parameter count and fast CPU execution using depthwise separable convolutions, squeeze-and-excitation channel attention, and lightweight activations (`HardSwish` / `HardSigmoid`).
* **ONNX Export & Runtime:** Models are exported to `.onnx` and executed locally via `onnxruntime.InferenceSession` on `CPUExecutionProvider`.
  * `mobilenetv3_small_fp32.onnx`: Full-precision (FP32) baseline for compatibility and reference accuracy.
  * `mobilenetv3_small_edge_final.onnx`: Half-precision (FP16) edge variant to reduce model size and memory footprint.

---


* **Key Takeaways to Fill In:**
  * Compare FP32 vs. FP16 accuracy trade-offs and file-size reduction on your test set.
  * Note hardware specs (CPU model, OS, Python / ONNX Runtime versions) used when measuring inference latency.

---

## Project Structure

```text
FarmEdge_Project/
├── app.py                              # Streamlit web UI, preprocessing, ONNX inference, and results display
├── README.md                           # Project documentation
└── models/
    ├── class_names.json                # Ordered list of 38 plant/disease class labels
    ├── mobilenetv3_small_fp32.onnx       # FP32 ONNX model
    └── mobilenetv3_small_edge_final.onnx # FP16 ONNX edge model


## Project files
- `app.py` — Streamlit entry point
- `requirements.txt` — Python dependencies
- `models/mobilenetv3_small_fp32.onnx` — recommended model for compatibility
- `models/mobilenetv3_small_edge_final.onnx` — optional smaller model
- `models/class_names.json` — class labels in model-output order

## Run on Windows (PowerShell / VS Code)
1. Extract this ZIP completely.
2. Open the extracted `FarmEdge_Project` folder in VS Code.
3. Check Python: `py --version`
4. Create a virtual environment: `py -m venv .venv`
5. Activate it: `.\.venv\Scripts\Activate.ps1`
   - If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and try activation again.
6. Install packages: `python -m pip install -r requirements.txt`
7. Start the app: `python -m streamlit run app.py`
8. Open the local URL printed in the terminal (usually `http://localhost:8501`).

## Notes and limitations
- Keep the `models` folder beside `app.py`; the app expects the model and label files at those paths.
- Start with the FP32 model selected in the sidebar.
- This app runs inference only; it does not train the model.
- Predictions may be inaccurate on external or field images, particularly when lighting, background, camera quality, or leaf orientation differs from the training data. The displayed model score is not a calibrated probability. Treat this as an academic prototype, not as the sole basis for crop-treatment decisions.
