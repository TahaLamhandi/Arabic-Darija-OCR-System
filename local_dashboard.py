import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw

import torch
import torch.nn as nn
import streamlit as st
import easyocr


class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.time_downsample = 4
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0),
            nn.ReLU(True),
        )
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True, dropout=0.1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        x = nn.functional.log_softmax(x, dim=2)
        return x


def preprocess_image_for_infer(img, img_height, max_width):
    img = ImageOps.grayscale(img)
    w, h = img.size
    new_w = int(img_height * w / h)
    new_w = max(1, min(new_w, max_width))
    img = img.resize((new_w, img_height), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return tensor


def greedy_decode(log_probs, idx_to_char, blank=0):
    max_indices = log_probs.argmax(2)
    results = []
    for n in range(max_indices.shape[1]):
        seq = max_indices[:, n].tolist()
        prev = None
        decoded = []
        for idx in seq:
            if idx != blank and idx != prev:
                decoded.append(idx_to_char.get(idx, ""))
            prev = idx
        results.append("".join(decoded))
    return results


@torch.no_grad()
def predict_pil(model, pil_img, idx_to_char_local, img_height, max_width, device):
    tensor = preprocess_image_for_infer(pil_img, img_height, max_width)
    log_probs = model(tensor.to(device))
    pred = greedy_decode(log_probs.detach().cpu(), idx_to_char_local, blank=0)[0]
    return pred


def resolve_default_model_path():
    candidates = [
        "best_darija_crnn.pth",
        "darija_crnn_last.pth",
        "darija_crnn_last.pkl",
    ]
    for name in candidates:
        if os.path.exists(name):
            return name
    return None


def load_model_from_path(model_path, device, fallback_chars=None):
    obj = torch.load(model_path, map_location=device)
    if isinstance(obj, dict) and "model_state" in obj:
        chars = obj.get("alphabet") or obj.get("charset") or obj.get("chars") or fallback_chars or []
        if isinstance(chars, str):
            chars = list(chars)
        idx_to_char_local = {i + 1: c for i, c in enumerate(chars)}
        model_local = CRNN(len(chars) + 1).to(device)
        model_local.load_state_dict(obj["model_state"])
        model_local.eval()
        cfg = {
            "img_height": obj.get("img_height", 32),
            "max_width": obj.get("max_width", 512),
        }
        return model_local, idx_to_char_local, cfg
    if isinstance(obj, nn.Module):
        model_local = obj.to(device)
        model_local.eval()
        cfg = {"img_height": 32, "max_width": 512}
        return model_local, {}, cfg
    raise ValueError("Unsupported model file")


@st.cache_resource(show_spinner=False)
def get_reader(lang_key, use_gpu):
    langs = ["ar"] if lang_key == "ar" else ["ar", "en"]
    return easyocr.Reader(langs, gpu=use_gpu, verbose=False)


@st.cache_resource(show_spinner=False)
def load_model_cached(model_path, mtime, device, fallback_chars):
    return load_model_from_path(model_path, device, fallback_chars=fallback_chars)


def save_uploaded_model(uploaded_file):
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def bbox_to_rect(bbox, w, h, pad_px=0):
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    x_min = max(0, int(min(xs) - pad_px))
    x_max = min(w, int(max(xs) + pad_px))
    y_min = max(0, int(min(ys) - pad_px))
    y_max = min(h, int(max(ys) + pad_px))
    return x_min, y_min, x_max, y_max


st.set_page_config(page_title="Darija OCR Dashboard", layout="wide")

css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&display=swap');
:root {
  --ink: #0b1f2a;
  --sea: #0ea5a8;
  --sand: #f5f1e6;
  --leaf: #2f855a;
  --shadow: rgba(15, 23, 42, 0.08);
}
html, body, [class*="css"] {
  font-family: 'Space Grotesk', sans-serif;
}
.stApp {
  background: radial-gradient(1200px 500px at 10% 5%, #e6fff6 0%, #f7f5ed 45%, #fff 100%);
}
.card {
  border: 1px solid #e8e4d8;
  border-radius: 14px;
  box-shadow: 0 10px 30px var(--shadow);
  padding: 14px;
  background: #ffffff;
}
.hero {
  animation: fadeIn 0.6s ease-out;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0px); }
}
</style>
"""

st.markdown(css, unsafe_allow_html=True)

st.markdown("<div class='hero'><h1>Darija OCR Dashboard</h1></div>", unsafe_allow_html=True)
st.caption("Local dashboard to test Arabic and Darija OCR using your trained model and EasyOCR detection.")

with st.sidebar:
    st.subheader("Model")
    model_upload = st.file_uploader("Upload model (.pth or .pkl)", type=["pth", "pkl"])
    model_path_input = st.text_input("Or enter model path")
    st.markdown("Note: Loading pickled models is unsafe for untrusted files.")

    st.subheader("Detector")
    lang_choice = st.selectbox("Detector languages", ["Arabic only", "Arabic + English"])
    min_conf = st.slider("Min detector confidence", 0.0, 1.0, value=0.35, step=0.01)
    max_boxes = st.slider("Max boxes", 1, 200, value=50, step=1)
    pad_px = st.slider("Crop padding (px)", 0, 20, value=2, step=1)
    final_source = st.radio("Final text source", ["Darija model", "EasyOCR"], index=0)


def resolve_model_path(model_upload, model_path_input):
    if model_upload is not None:
        if st.session_state.get("uploaded_model_name") != model_upload.name:
            st.session_state["uploaded_model_name"] = model_upload.name
            st.session_state["uploaded_model_path"] = save_uploaded_model(model_upload)
        return st.session_state.get("uploaded_model_path")
    if model_path_input and os.path.exists(model_path_input):
        return model_path_input
    return resolve_default_model_path()


def run_ocr(image, model_path, lang_key, min_conf, max_boxes, pad_px, final_source):
    if image is None:
        empty_df = pd.DataFrame(columns=["idx", "easyocr_ar", "conf", "darija_recognizer"])
        return None, "", "", empty_df

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fallback_chars = []
    if model_path is None:
        raise RuntimeError("Model not found. Upload a .pth/.pkl or set a valid path.")

    mtime = os.path.getmtime(model_path)
    model, idx_to_char, cfg = load_model_cached(model_path, mtime, device, fallback_chars)
    if not idx_to_char:
        raise RuntimeError("Charset not found in model file.")

    reader = get_reader(lang_key, use_gpu=(device.type == "cuda"))

    pil_img = image.convert("RGB")
    np_img = np.array(pil_img)
    results = reader.readtext(np_img, detail=1, paragraph=False)
    results = sorted(results, key=lambda r: (min(p[1] for p in r[0]), min(p[0] for p in r[0])))

    img_h = cfg.get("img_height", 32)
    max_w = cfg.get("max_width", 512)
    pad_px = int(pad_px)

    rows = []
    draw = ImageDraw.Draw(pil_img)
    for _, (bbox, easy_text, conf) in enumerate(results):
        if conf < min_conf:
            continue
        if len(rows) >= int(max_boxes):
            break
        x_min, y_min, x_max, y_max = bbox_to_rect(bbox, pil_img.width, pil_img.height, pad_px=pad_px)
        if x_max <= x_min or y_max <= y_min:
            continue
        crop = pil_img.crop((x_min, y_min, x_max, y_max))
        darija_text = predict_pil(model, crop, idx_to_char, img_h, max_w, device)
        rows.append([len(rows), easy_text, float(conf), darija_text])
        draw.rectangle([x_min, y_min, x_max, y_max], outline=(0, 210, 140), width=2)

    if len(rows) == 0:
        darija_text = predict_pil(model, pil_img, idx_to_char, img_h, max_w, device)
        rows = [[0, "", 0.0, darija_text]]
        draw.rectangle([0, 0, pil_img.width - 1, pil_img.height - 1], outline=(0, 210, 140), width=2)

    df_out = pd.DataFrame(rows, columns=["idx", "easyocr_ar", "conf", "darija_recognizer"])
    darija_full = "\n".join(df_out["darija_recognizer"].tolist())
    easy_full = "\n".join(df_out["easyocr_ar"].tolist())
    final_text = darija_full if final_source == "Darija model" else easy_full
    return pil_img, final_text, easy_full, df_out


left, right = st.columns(2, gap="large")

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    image = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"])
    run = st.button("Run OCR", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    out_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

model_path = resolve_model_path(model_upload, model_path_input)
lang_key = "ar" if lang_choice == "Arabic only" else "ar_en"

if run:
    if image is None:
        st.warning("Please upload an image.")
    else:
        pil_img = Image.open(image)
        try:
            annotated, final_text, easy_text, df_out = run_ocr(
                pil_img, model_path, lang_key, min_conf, max_boxes, pad_px, final_source
            )
        except Exception as exc:
            st.error(f"OCR failed: {exc}")
        else:
            out_placeholder.image(annotated, caption="Detections", use_column_width=True)
            st.subheader("Final text")
            st.text_area("", final_text, height=140, key="final_text_area")
            st.subheader("EasyOCR text")
            st.text_area("", easy_text, height=140, key="easy_text_area")
            st.subheader("Per-box results")
            st.dataframe(df_out, use_container_width=True)

if model_path:
    st.caption(f"Using model: {model_path}")
else:
    st.caption("No model selected. Upload a model or set a valid path in the sidebar.")
