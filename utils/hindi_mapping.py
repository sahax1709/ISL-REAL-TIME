"""English to Hindi/Devanagari mappings."""

LETTER_TO_HINDI = {
    "A": "\u0905", "B": "\u092c", "C": "\u0938", "D": "\u0921", "E": "\u0907",
    "F": "\u092b", "G": "\u0917", "H": "\u0939", "I": "\u0908", "J": "\u091c",
    "K": "\u0915", "L": "\u0932", "M": "\u092e", "N": "\u0928", "O": "\u0913",
    "P": "\u092a", "Q": "\u0915\u093c", "R": "\u0930", "S": "\u0938", "T": "\u091f",
    "U": "\u0909", "V": "\u0935", "W": "\u0935\u093c", "X": "\u090f\u0915\u094d\u0938",
    "Y": "\u092f", "Z": "\u091c\u093c",
}

DIGIT_TO_HINDI = {
    "0": "\u0966", "1": "\u0967", "2": "\u0968", "3": "\u0969", "4": "\u096a",
    "5": "\u096b", "6": "\u096c", "7": "\u096d", "8": "\u096e", "9": "\u096f",
}

WORD_TO_HINDI = {
    "hello": "\u0928\u092e\u0938\u094d\u0924\u0947",
    "thank_you": "\u0927\u0928\u094d\u092f\u0935\u093e\u0926",
    "please": "\u0915\u0943\u092a\u092f\u093e",
    "sorry": "\u092e\u093e\u092b\u093c \u0915\u0940\u091c\u093f\u090f",
    "yes": "\u0939\u093e\u0901",
    "no": "\u0928\u0939\u0940\u0902",
    "help": "\u092e\u0926\u0926",
    "stop": "\u0930\u0941\u0915\u094b",
    "good": "\u0905\u091a\u094d\u091b\u093e",
    "bad": "\u092c\u0941\u0930\u093e",
    "love": "\u092a\u094d\u092f\u093e\u0930",
    "friend": "\u0926\u094b\u0938\u094d\u0924",
    "family": "\u092a\u0930\u093f\u0935\u093e\u0930",
    "eat": "\u0916\u093e\u0928\u093e",
    "drink": "\u092a\u0940\u0928\u093e",
    "water": "\u092a\u093e\u0928\u0940",
    "more": "\u0914\u0930",
    "done": "\u0939\u094b \u0917\u092f\u093e",
    "want": "\u091a\u093e\u0939\u093f\u090f",
    "need": "\u091c\u093c\u0930\u0942\u0930\u0924",
    "go": "\u091c\u093e\u0913",
    "come": "\u0906\u0913",
    "home": "\u0918\u0930",
    "school": "\u0938\u094d\u0915\u0942\u0932",
    "work": "\u0915\u093e\u092e",
    "name": "\u0928\u093e\u092e",
    "how": "\u0915\u0948\u0938\u0947",
    "what": "\u0915\u094d\u092f\u093e",
    "where": "\u0915\u0939\u093e\u0901",
    "when": "\u0915\u092c",
}

ALL_CLASSES = (
    sorted(LETTER_TO_HINDI.keys())
    + sorted(DIGIT_TO_HINDI.keys())
    + sorted(WORD_TO_HINDI.keys())
)

CLASS_TO_INDEX = {c: i for i, c in enumerate(ALL_CLASSES)}
INDEX_TO_CLASS = {i: c for c, i in CLASS_TO_INDEX.items()}
NUM_CLASSES = len(ALL_CLASSES)


def to_hindi(label):
    up = label.upper()
    if up in LETTER_TO_HINDI:
        return LETTER_TO_HINDI[up]
    if label in DIGIT_TO_HINDI:
        return DIGIT_TO_HINDI[label]
    lo = label.lower()
    if lo in WORD_TO_HINDI:
        return WORD_TO_HINDI[lo]
    return label


def to_display(label):
    label = label.strip()
    is_letter = label.upper() in LETTER_TO_HINDI
    is_digit = label in DIGIT_TO_HINDI
    is_word = label.lower() in WORD_TO_HINDI
    cat = "letter" if is_letter else "digit" if is_digit else "word" if is_word else "unknown"
    return {
        "english": label.upper() if is_letter else label.capitalize() if is_word else label,
        "hindi": to_hindi(label),
        "category": cat,
    }
