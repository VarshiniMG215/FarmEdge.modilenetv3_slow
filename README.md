# FarmEdge — Lightweight Plant Disease Detection

FarmEdge is a Streamlit prototype that classifies an uploaded plant-leaf image using a MobileNetV3-Small model exported to ONNX. It displays the predicted class, model score, inference time, and top-three classes.

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
