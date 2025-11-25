# src/postprocess.py
import re

#########################################################
# COMMON HELPERS
#########################################################

WORD_DIGIT = {
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9"
}

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec",
          "january", "february", "march", "april", "june",
          "july", "august", "september", "october",
          "november", "december"]

CITY_HINTS = [
    "delhi", "mumbai", "pune", "surat", "lucknow", "kolkata",
    "bangalore", "hyderabad", "indore", "jaipur", "coimbatore",
    "kochi", "nagpur", "gurgaon", "ahmedabad", "trivandrum"
]

def digits_only(s: str) -> str:
    return re.sub(r"\D", "", s)


#########################################################
# CREDIT CARD (STRICT P0)
#########################################################

def normalize_creditcard(candidate: str) -> str:
    tokens = candidate.lower().split()
    out = []
    i = 0

    while i < len(tokens):
        tk = tokens[i]

        # handle "double" or "triple"
        if tk in ("double", "triple") and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt in WORD_DIGIT:
                d = WORD_DIGIT[nxt]
                out.append(d)
                if tk == "double":
                    out.append(d)
                else:  # triple
                    out.extend([d, d])
            i += 2
            continue

        if tk in WORD_DIGIT:
            out.append(WORD_DIGIT[tk])
        elif tk.isdigit():
            out.append(tk)

        i += 1

    return digits_only("".join(out))


def is_valid_credit_card(candidate: str) -> bool:
    num = normalize_creditcard(candidate)
    # Do not enforce Luhn for spoken cards.
    return 13 <= len(num) <= 19


#########################################################
# PHONE (STRICT P0)
#########################################################

#########################################################
# PHONE (STRICT P0 — improved)
#########################################################

def normalize_phone(candidate: str) -> str:
    tokens = candidate.lower().split()
    out = []
    i = 0

    while i < len(tokens):
        tk = tokens[i]

        # handle "double" / "triple"
        if tk in ("double", "triple") and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            # allow spoken digits
            if nxt in WORD_DIGIT:
                d = WORD_DIGIT[nxt]
                if tk == "double":
                    out.extend([d, d])
                else:
                    out.extend([d, d, d])
                i += 2
                continue

        # spoken digits
        if tk in WORD_DIGIT:
            out.append(WORD_DIGIT[tk])

        # numeric chunks
        elif tk.isdigit():
            out.append(tk)

        # handle "oh" / "o" as zero
        elif tk == "oh" or tk == "o":
            out.append("0")

        i += 1

    return digits_only("".join(out))


def is_valid_phone(candidate: str) -> bool:
    num = normalize_phone(candidate)

    # Reject if number looks like credit card
    if len(num) >= 13:
        return False

    # Valid phone numbers: 7–12 digits
    return 7 <= len(num) <= 12



#########################################################
# EMAIL (STRICT P0)
#########################################################

def collapse_spaced_letters(s: str) -> str:
    # Convert: "h o t m a i l" -> "hotmail"
    return re.sub(
        r"(?:\b[a-zA-Z]\b\s*){2,}",
        lambda m: "".join(m.group(0).split()),
        s
    )


def normalize_email_candidate(s: str) -> str:
    s = s.strip().lower()

    # collapse spaced letters first
    s = collapse_spaced_letters(s)

    # spoken dot
    s = re.sub(r"\bdot\b", ".", s)
    s = re.sub(r"\bdott\b", ".", s)

    # spoken at
    s = re.sub(r"\bat\b", "@", s)

    return s.replace(" ", "")


EMAIL_REGEX = re.compile(
    r"^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,5}$"
)


def is_valid_email(candidate: str) -> bool:
    norm = normalize_email_candidate(candidate)
    return bool(EMAIL_REGEX.match(norm))


#########################################################
# PERSON NAME (STRICT P0 + PN2)
#########################################################

# Hard stopwords that must NOT appear inside a name span:
PN_FORBIDDEN = {
    "haan", "so", "my", "naam", "is", "and", "main", "rehte", "in",
    "uh", "actually", "old", "card", "number", "maybe", "please",
    "send", "email", "phone", "today", "tomorrow", "from"
}

# Spoken digit words must reject PERSON_NAME:
SPOKEN_NUM_WORDS = set(WORD_DIGIT.keys()).union({"double", "triple"})

def is_valid_person_name(candidate: str) -> bool:
    s = candidate.strip().lower()
    tokens = s.split()

    # 1–2 tokens only (PN2 strict mode)
    if not (1 <= len(tokens) <= 2):
        return False

    # Reject digits anywhere
    if any(ch.isdigit() for ch in s):
        return False

    # Reject tokens shorter than 2 chars (removes spaced-letter garbage)
    if any(len(t) < 2 for t in tokens):
        return False

    # Reject if any forbidden token appears
    if any(t in PN_FORBIDDEN for t in tokens):
        return False

    # Reject if spoken-digit words are inside
    if any(t in SPOKEN_NUM_WORDS for t in tokens):
        return False

    # Reject if token matches city names or months (avoid false positives)
    if any(t in CITY_HINTS for t in tokens):
        return False
    if any(t in MONTHS for t in tokens):
        return False

    # Reject if it contains email-like markers
    if "dot" in s or "@" in s:
        return False

    # All tokens must be alphabetic words
    if not all(t.isalpha() for t in tokens):
        return False

    return True


#########################################################
# DATE (unchanged)
#########################################################

def is_valid_date(candidate: str) -> bool:
    s = candidate.lower()

    if any(ch.isdigit() for ch in s) and len(s) <= 40:
        return True

    if any(m in s for m in MONTHS):
        return True

    return False


#########################################################
# MAIN FILTER
#########################################################

def filter_spans(spans, text):
    out = []
    for s, e, lab in spans:
        cand = text[s:e]

        if lab == "CREDIT_CARD":
            keep = is_valid_credit_card(cand)
        elif lab == "PHONE":
            keep = is_valid_phone(cand)
        elif lab == "EMAIL":
            keep = is_valid_email(cand)
        elif lab == "PERSON_NAME":
            keep = is_valid_person_name(cand)
        elif lab == "DATE":
            keep = is_valid_date(cand)
        else:
            keep = True

        if keep:
            out.append({"start": s, "end": e, "label": lab})

    return out