from ocr import extract_text
from correction import normalize_arabic

image_paths = [
    "images/test.png",
    "images/test2.png"
]

for image_path in image_paths:

    print("\n" + "=" * 60)
    print(f"Processing: {image_path}")
    print("=" * 60)

    raw_text = extract_text(image_path)

    corrected_text = normalize_arabic(raw_text)

    print("\nRAW OCR:")
    print(raw_text)

    print("\nNORMALIZED:")
    print(corrected_text)