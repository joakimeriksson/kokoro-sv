"""Swedish text normalization for TTS.

Design rules (from the project plan):
  * numbers -> words ("42" -> "fyrtiotvå")
  * dates   -> words ("2026-07-06" -> "sjätte juli tjugohundratjugosex")
  * times   -> words ("14:30" -> "fjorton och trettio")
  * common abbreviations expanded ("t.ex." -> "till exempel")
  * English/technical tokens are KEPT verbatim (the G2P handles them)
  * punctuation that carries prosody is PRESERVED: . , ! ? — :
  * junk characters removed
"""
from __future__ import annotations

import re

ONES = ["noll", "ett", "två", "tre", "fyra", "fem", "sex", "sju", "åtta", "nio",
        "tio", "elva", "tolv", "tretton", "fjorton", "femton", "sexton",
        "sjutton", "arton", "nitton"]
TENS = ["", "", "tjugo", "trettio", "fyrtio", "femtio", "sextio", "sjuttio",
        "åttio", "nittio"]
ORDINALS = {1: "första", 2: "andra", 3: "tredje", 4: "fjärde", 5: "femte",
            6: "sjätte", 7: "sjunde", 8: "åttonde", 9: "nionde", 10: "tionde",
            11: "elfte", 12: "tolfte", 13: "trettonde", 14: "fjortonde",
            15: "femtonde", 16: "sextonde", 17: "sjuttonde", 18: "artonde",
            19: "nittonde", 20: "tjugonde", 30: "trettionde", 31: "trettioförsta"}
MONTHS = ["", "januari", "februari", "mars", "april", "maj", "juni", "juli",
          "augusti", "september", "oktober", "november", "december"]

ABBREV = {
    "t.ex.": "till exempel", "t ex": "till exempel", "bl.a.": "bland annat",
    "dvs.": "det vill säga", "d.v.s.": "det vill säga", "osv.": "och så vidare",
    "o.s.v.": "och så vidare", "m.m.": "med mera", "mm.": "med mera",
    "ca": "cirka", "ca.": "cirka", "st.": "stycken", "st": "stycken",
    "kr": "kronor", "nr": "nummer", "nr.": "nummer", "kl.": "klockan",
    "kl": "klockan", "m.fl.": "med flera", "etc.": "et cetera",
    "resp.": "respektive", "inkl.": "inklusive", "exkl.": "exklusive",
}


def int_to_words(n: int) -> str:
    """Swedish cardinal words, supports 0 .. 999_999_999."""
    if n < 0:
        return "minus " + int_to_words(-n)
    if n < 20:
        return ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return TENS[t] + (ONES[o] if o else "")
    if n < 1000:
        h, r = divmod(n, 100)
        pre = ("ett" if h == 1 else ONES[h]) + "hundra"
        return pre + (int_to_words(r) if r else "")
    if n < 1_000_000:
        k, r = divmod(n, 1000)
        pre = "ettusen" if k == 1 else int_to_words(k) + "tusen"
        return pre + (" " + int_to_words(r) if r else "")
    m, r = divmod(n, 1_000_000)
    pre = ("en miljon" if m == 1 else int_to_words(m) + " miljoner")
    return pre + (" " + int_to_words(r) if r else "")


def ordinal_to_words(n: int) -> str:
    if n in ORDINALS:
        return ORDINALS[n]
    if 20 < n < 30:
        return "tjugo" + ORDINALS[n - 20]
    return int_to_words(n)  # fallback: cardinal


def year_to_words(y: int) -> str:
    if 1100 <= y < 2000:
        h, r = divmod(y, 100)
        return int_to_words(h) + "hundra" + (int_to_words(r) if r else "")
    if 2000 <= y < 2100:
        r = y - 2000
        return "tjugohundra" + (int_to_words(r) if r else "")
    return int_to_words(y)


def _date(m: re.Match) -> str:
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return m.group(0)
    return f"{ordinal_to_words(d)} {MONTHS[mo]} {year_to_words(y)}"


def _time(m: re.Match) -> str:
    h, mi = int(m.group(1)), int(m.group(2))
    if not (0 <= h <= 23 and 0 <= mi <= 59):
        return m.group(0)
    if mi == 0:
        return int_to_words(h)
    return f"{int_to_words(h)} och {int_to_words(mi)}"


def _decimal(m: re.Match) -> str:
    whole, frac = m.group(1), m.group(2)
    return f"{int_to_words(int(whole))} komma {' '.join(ONES[int(c)] for c in frac)}"


def _plain_int(m: re.Match) -> str:
    s = m.group(0).replace(" ", "").replace(".", "")
    try:
        return int_to_words(int(s))
    except ValueError:
        return m.group(0)


def normalize_text_sv(text: str) -> str:
    t = " " + text.strip() + " "

    # abbreviations (longest first, word-ish boundaries)
    for abbr in sorted(ABBREV, key=len, reverse=True):
        t = re.sub(rf"(?<=[\s(]){re.escape(abbr)}(?=[\s.,!?)])", ABBREV[abbr], t)

    t = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", _date, t)          # ISO dates
    t = re.sub(r"\b(\d{1,2})[:.](\d{2})\b", _time, t)             # clock times
    t = re.sub(r"\b(\d+)[,.](\d+)\b", _decimal, t)                # decimals
    t = re.sub(r"\b\d{1,3}(?: \d{3})+\b", _plain_int, t)          # 1 000 000
    t = re.sub(r"\b\d+\b", _plain_int, t)                         # plain ints

    t = re.sub(r"%", " procent", t)
    t = re.sub(r"&", " och ", t)
    t = re.sub(r"[\"“”„«»]", "", t)
    t = re.sub(r"[*_#@^~|<>{}\[\]\\/=+]", " ", t)                 # junk chars
    t = re.sub(r"\s*—\s*", " — ", t)                              # keep em-dash
    t = re.sub(r"\s+([.,!?:])", r"\1", t)                         # tidy spacing
    t = re.sub(r"\s+", " ", t).strip()
    return t
