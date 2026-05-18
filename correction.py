import re

def normalize_arabic(text):

    if not text:
        return text

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    # Normalize alef only
    text = re.sub(r'[أإآ]', 'ا', text)

    # Remove tatweel
    text = re.sub(r'ـ', '', text)

    return text.strip()