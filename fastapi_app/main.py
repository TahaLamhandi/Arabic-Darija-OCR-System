import base64
import io
import os
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps, UnidentifiedImageError

import torch
import torch.nn as nn
import easyocr

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


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
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Sharpness(img).enhance(1.15)
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


MAX_UPLOAD_MB = 8
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
    "image/tiff",
}
Image.MAX_IMAGE_PIXELS = 20_000_000


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


def load_model_from_path(model_path, device):
    obj = torch.load(model_path, map_location=device)
    if isinstance(obj, dict) and "model_state" in obj:
        chars = obj.get("alphabet") or obj.get("charset") or obj.get("chars") or []
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


def validate_upload(file: UploadFile, content: bytes) -> Tuple[bool, str]:
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_MIME_TYPES:
        return False, "Unsupported file type. Please upload PNG, JPG, WEBP, BMP, GIF, or TIFF."
    if not content:
        return False, "Empty file. Please upload a valid image."
    if len(content) > MAX_UPLOAD_BYTES:
        return False, f"File too large. Max size is {MAX_UPLOAD_MB} MB."
    return True, ""


def bbox_to_rect(bbox, w, h, pad_px=2):
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    x_min = max(0, int(min(xs) - pad_px))
    x_max = min(w, int(max(xs) + pad_px))
    y_min = max(0, int(min(ys) - pad_px))
    y_max = min(h, int(max(ys) + pad_px))
    return x_min, y_min, x_max, y_max


def pil_to_data_uri(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def has_arabic(text):
    for ch in text:
        if "\u0600" <= ch <= "\u06FF" or "\u0750" <= ch <= "\u077F" or "\u08A0" <= ch <= "\u08FF":
            return True
    return False


def scale_bbox(bbox, scale):
    return [[p[0] / scale, p[1] / scale] for p in bbox]


def enhance_for_easyocr(pil_img, scale=2):
    img = pil_img.convert("RGB")
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.BICUBIC)
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Sharpness(img).enhance(1.2)
    img = ImageEnhance.Contrast(img).enhance(1.15)
    return img.convert("RGB"), scale


def results_quality(results):
    if not results:
        return 0.0, 0
    avg_conf = sum(r[2] for r in results) / len(results)
    text_len = sum(len(r[1]) for r in results if r[1])
    return avg_conf, text_len


def order_results_and_join(results, img_height, rtl=False):
    if not results:
        return [], ""
    y_thresh = max(10, int(img_height * 0.06))
    base_sorted = sorted(results, key=lambda r: (min(p[1] for p in r[0]), min(p[0] for p in r[0])))
    lines = []
    for item in base_sorted:
        y = min(p[1] for p in item[0])
        if not lines or abs(y - lines[-1][0]) > y_thresh:
            lines.append([y, [item]])
        else:
            lines[-1][1].append(item)

    ordered = []
    line_texts = []
    for _, items in lines:
        if rtl:
            items = sorted(items, key=lambda r: -max(p[0] for p in r[0]))
        else:
            items = sorted(items, key=lambda r: min(p[0] for p in r[0]))
        ordered.extend(items)
        line_text = " ".join([r[1] for r in items if r[1]])
        if line_text:
            line_texts.append(line_text)
    return ordered, "\n".join(line_texts)


def group_results_by_line(results, img_height):
    if not results:
        return []
    y_thresh = max(10, int(img_height * 0.06))
    base_sorted = sorted(results, key=lambda r: (min(p[1] for p in r[0]), min(p[0] for p in r[0])))
    lines = []
    for item in base_sorted:
        y = min(p[1] for p in item[0])
        if not lines or abs(y - lines[-1][0]) > y_thresh:
            lines.append([y, [item]])
        else:
            lines[-1][1].append(item)
    return [line_items for _, line_items in lines]


def line_bbox_to_rect(line_items, w, h, pad_ratio=0.08, min_pad=4, max_pad=24):
    xs = []
    ys = []
    for bbox, _, _ in line_items:
        xs.extend([p[0] for p in bbox])
        ys.extend([p[1] for p in bbox])
    if not xs or not ys:
        return 0, 0, w, h
    line_h = max(1, int(max(ys) - min(ys)))
    pad_px = int(max(min_pad, min(max_pad, line_h * pad_ratio)))
    x_min = max(0, int(min(xs) - pad_px))
    x_max = min(w, int(max(xs) + pad_px))
    y_min = max(0, int(min(ys) - pad_px))
    y_max = min(h, int(max(ys) + pad_px))
    return x_min, y_min, x_max, y_max


def run_ocr(image, reader, model, idx_to_char, cfg, device):
    pil_img = image.convert("RGB")
    np_img = np.array(pil_img)
    results = reader.readtext(np_img, detail=1, paragraph=False)

    enhanced_img, scale = enhance_for_easyocr(pil_img, scale=2)
    enhanced_results = reader.readtext(np.array(enhanced_img), detail=1, paragraph=False)
    if scale != 1:
        enhanced_results = [(scale_bbox(b, scale), t, c) for b, t, c in enhanced_results]

    q1 = results_quality(results)
    q2 = results_quality(enhanced_results)
    if q2[0] > q1[0] + 0.02 or (q2[0] >= q1[0] and q2[1] > q1[1]):
        results = enhanced_results

    joined_text = " ".join([r[1] for r in results if r[1]])
    rtl = has_arabic(joined_text)
    ordered_results, easy_full = order_results_and_join(results, pil_img.height, rtl=rtl)
    lines = group_results_by_line(results, pil_img.height)

    img_h = cfg.get("img_height", 32)
    max_w = cfg.get("max_width", 512)

    rows = []
    draw = ImageDraw.Draw(pil_img)
    for bbox, easy_text, conf in ordered_results:
        x_min, y_min, x_max, y_max = bbox_to_rect(bbox, pil_img.width, pil_img.height)
        if x_max <= x_min or y_max <= y_min:
            continue
        rows.append((len(rows), easy_text, float(conf)))
        draw.rectangle([x_min, y_min, x_max, y_max], outline=(0, 210, 140), width=2)

    darija_lines = []
    for line_items in lines:
        x_min, y_min, x_max, y_max = line_bbox_to_rect(line_items, pil_img.width, pil_img.height)
        if x_max <= x_min or y_max <= y_min:
            continue
        crop = pil_img.crop((x_min, y_min, x_max, y_max))
        darija_text = predict_pil(model, crop, idx_to_char, img_h, max_w, device)
        if darija_text:
            darija_lines.append(darija_text)

    darija_full = "\n".join(darija_lines)
    if darija_full.strip() == "":
        darija_full = predict_pil(model, pil_img, idx_to_char, img_h, max_w, device)
    return pil_img, rows, darija_full, easy_full


def build_app():
    app = FastAPI()
    app.mount("/static", StaticFiles(directory="fastapi_app/static"), name="static")
    templates = Jinja2Templates(directory="fastapi_app/templates")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.environ.get("DARIJA_MODEL_PATH") or resolve_default_model_path()

    model = None
    idx_to_char = {}
    cfg = {"img_height": 32, "max_width": 512}
    model_status = "Modele non charge"

    if model_path:
        try:
            model, idx_to_char, cfg = load_model_from_path(model_path, device)
            if not idx_to_char:
                model_status = f"Modele charge mais charset manquant: {model_path}"
            else:
                model_status = f"Modele charge: {model_path}"
        except Exception as exc:
            model_status = f"Erreur de chargement du modele: {exc}"
    else:
        model_status = "Aucun modele trouve dans le dossier"

    reader = easyocr.Reader(["ar", "en"], gpu=(device.type == "cuda"), verbose=False)

    def render_template(request: Request, context: dict):
        import inspect

        sig = inspect.signature(templates.TemplateResponse)
        params = list(sig.parameters.values())
        if params and params[0].name == "request":
            return templates.TemplateResponse(request, "index.html", context)
        return templates.TemplateResponse("index.html", context)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return render_template(
            request,
            {
                "request": request,
                "model_status": model_status,
                "results": None,
            },
        )

    @app.post("/ocr", response_class=HTMLResponse)
    async def ocr(request: Request, file: UploadFile = File(...)):
        if model is None or not idx_to_char:
            return render_template(
                request,
                {
                    "request": request,
                    "model_status": model_status,
                    "results": None,
                    "error": "Modele indisponible ou charset manquant.",
                },
            )

        content = await file.read()
        valid, error = validate_upload(file, content)
        if not valid:
            return render_template(
                request,
                {
                    "request": request,
                    "model_status": model_status,
                    "results": None,
                    "error": error,
                },
            )

        try:
            image = Image.open(io.BytesIO(content))
            image.load()
            image = image.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            return render_template(
                request,
                {
                    "request": request,
                    "model_status": model_status,
                    "results": None,
                    "error": "Invalid or unsupported image file.",
                },
            )

        annotated, rows, darija_text, easy_text = run_ocr(image, reader, model, idx_to_char, cfg, device)
        data_uri = pil_to_data_uri(annotated)

        return render_template(
            request,
            {
                "request": request,
                "model_status": model_status,
                "results": {
                    "image": data_uri,
                    "darija_text": darija_text,
                    "easy_text": easy_text,
                    "rows": rows,
                    "filename": file.filename,
                },
            },
        )

    return app


app = build_app()
