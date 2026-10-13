"""Strip the sentences two texts share so the verifier only reads what changed.

A long prompt that differs in one line (same passage, different question) makes an
NLI model see near-total overlap and saturate entailment near 1.0 for paraphrases
and traps alike. Comparing only the differing sentences turns that into a short
pair the model can actually judge.

Only sentences that are identical after case and whitespace folding are removed.
Deciding that two differently worded sentences "mean the same" would need a model,
and a wrong call there would hide the very difference the verifier must see. When
nothing is shared the texts are returned unchanged, so the failure mode is "no
reduction", never "difference hidden".

Must stay identical to the offline experiment (``diffpair.reduce_pair`` with
``normalize=True``) the cache threshold was measured with.
"""

from __future__ import annotations

import difflib
import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"\s+")


def _segments(text: str) -> list[str]:
    out: list[str] = []
    for line in text.split("\n"):
        for sentence in _SENTENCE_END.split(line.strip()):
            sentence = sentence.strip()
            if sentence:
                out.append(sentence)
    return out


def _key(sentence: str) -> str:
    # Case and whitespace only. Digits, symbols and punctuation are left alone:
    # folding any of them could make "120 books" and "150 books" look shared.
    return _WHITESPACE.sub(" ", sentence).strip().lower()


def reduce_pair(a: str, b: str) -> tuple[str, str]:
    """Return the sentences of ``a`` and ``b`` that the other text does not share.

    1. Split on newlines, then after sentence-ending punctuation.
    2. Align the two sentence lists with difflib and keep the non-equal spans.
    3. If one side ends up empty (a pure insertion or deletion), prepend the shared
       sentence just before each change (or just after, at the start) to both sides
       so neither input is blank.
    4. If no sentence is shared, return the originals.
    """
    sa, sb = _segments(a), _segments(b)
    ka, kb = [_key(s) for s in sa], [_key(s) for s in sb]
    ops = difflib.SequenceMatcher(None, ka, kb, autojunk=False).get_opcodes()
    if not any(tag == "equal" for tag, *_ in ops):
        return a, b

    da: list[str] = []
    db: list[str] = []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        da.extend(sa[i1:i2])
        db.extend(sb[j1:j2])

    if not da or not db:
        context: list[str] = []
        for tag, i1, i2, _j1, _j2 in ops:
            if tag == "equal":
                continue
            if i1 > 0:
                context.append(sa[i1 - 1])
            elif i2 < len(sa):
                context.append(sa[i2])
        da, db = context + da, context + db

    if not da or not db:
        return a, b
    return " ".join(da), " ".join(db)
