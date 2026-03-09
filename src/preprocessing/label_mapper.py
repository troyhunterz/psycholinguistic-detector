LABEL_MAP = {
    # Propaganda
    "appeal_to_fear":       "fear_appeal",
    "appeal_to_authority":  "authority_appeal",
    "loaded_language":      "emotional_manipulation",
    "black-and-white":      "demagogy_tricks",
    "casual_oversimplif":   "demagogy_tricks",
    "repetition":           "demagogy_tricks",

    # Fake news
    "fake":         "emotional_manipulation",
    "real":         "rational_argument",

    # Argument mining
    "support":          "rational_argument",
    "attack":           "demagogy_tricks",

    # Hate speech
    "hate":         "emotional_manipulation",
    "neutral":      "rational_argument",
}


def map_label(original_label: str) -> str:
    return LABEL_MAP.get(original_label.lower(), None)
