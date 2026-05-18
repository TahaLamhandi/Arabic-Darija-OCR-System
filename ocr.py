import cv2
import easyocr
import re

print("Loading EasyOCR Arabic model...")

reader = easyocr.Reader(
    ['ar'],
    gpu=False,
    verbose=False
)

print("Model loaded!")

# ============================================================
# SIMPLE PREPROCESSING
# ============================================================

def preprocess_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise Exception(f"Cannot load image: {image_path}")

    # Convert to grayscale only
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Very light denoise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return gray

# ============================================================
# OCR EXTRACTION
# ============================================================

def extract_text(image_path):

    processed = preprocess_image(image_path)

    result = reader.readtext(
        processed,
        detail=0,
        paragraph=True,
        contrast_ths=0.3,
        adjust_contrast=0.5,
        width_ths=0.7
    )

    text = " ".join(result)

    # Clean spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()