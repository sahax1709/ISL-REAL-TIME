"""
English to Hindi/Devanagari mappings for sign language output.
- Letters: English → Devanagari phonetic equivalents
- Digits: Arabic numerals → Devanagari numerals
- Words: English → Hindi translations
"""

# ── A-Z → Devanagari phonetic letter equivalents ──
LETTER_TO_HINDI = {
    "A": "अ", "B": "ब", "C": "स", "D": "ड", "E": "इ",
    "F": "फ", "G": "ग", "H": "ह", "I": "ई", "J": "ज",
    "K": "क", "L": "ल", "M": "म", "N": "न", "O": "ओ",
    "P": "प", "Q": "क़", "R": "र", "S": "स", "T": "ट",
    "U": "उ", "V": "व", "W": "व़", "X": "एक्स", "Y": "य",
    "Z": "ज़",
}

# ── 0-9 → Devanagari numerals ──
DIGIT_TO_HINDI = {
    "0": "०", "1": "१", "2": "२", "3": "३", "4": "४",
    "5": "५", "6": "६", "7": "७", "8": "८", "9": "९",
}

# ── Common sign language words → Hindi translations ──
WORD_TO_HINDI = {
    "hello":      "नमस्ते",
    "thank_you":  "धन्यवाद",
    "please":     "कृपया",
    "sorry":      "माफ़ कीजिए",
    "yes":        "हाँ",
    "no":         "नहीं",
    "help":       "मदद",
    "stop":       "रुको",
    "good":       "अच्छा",
    "bad":        "बुरा",
    "love":       "प्यार",
    "friend":     "दोस्त",
    "family":     "परिवार",
    "eat":        "खाना",
    "drink":      "पीना",
    "water":      "पानी",
    "more":       "और",
    "done":       "हो गया",
    "want":       "चाहिए",
    "need":       "ज़रूरत",
    "go":         "जाओ",
    "come":       "आओ",
    "home":       "घर",
    "school":     "स्कूल",
    "work":       "काम",
    "name":       "नाम",
    "how":        "कैसे",
    "what":       "क्या",
    "where":      "कहाँ",
    "when":       "कब",
}

# ── Build unified class list ──
ALL_CLASSES = (
    sorted(LETTER_TO_HINDI.keys())
    + sorted(DIGIT_TO_HINDI.keys())
    + sorted(WORD_TO_HINDI.keys())
)

CLASS_TO_INDEX = {cls: idx for idx, cls in enumerate(ALL_CLASSES)}
INDEX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_INDEX.items()}
NUM_CLASSES = len(ALL_CLASSES)


def to_hindi(label: str) -> str:
    """Convert an English label to its Hindi equivalent."""
    label_upper = label.upper()
    if label_upper in LETTER_TO_HINDI:
        return LETTER_TO_HINDI[label_upper]
    if label in DIGIT_TO_HINDI:
        return DIGIT_TO_HINDI[label]
    label_lower = label.lower()
    if label_lower in WORD_TO_HINDI:
        return WORD_TO_HINDI[label_lower]
    return label


def to_display(label: str) -> dict:
    """Return a display-ready dict with English and Hindi."""
    label_clean = label.strip()
    is_letter = label_clean.upper() in LETTER_TO_HINDI
    is_digit = label_clean in DIGIT_TO_HINDI
    is_word = label_clean.lower() in WORD_TO_HINDI

    category = "letter" if is_letter else "digit" if is_digit else "word" if is_word else "unknown"
    return {
        "english": label_clean.upper() if is_letter else label_clean.capitalize() if is_word else label_clean,
        "hindi": to_hindi(label_clean),
        "category": category,
    }
