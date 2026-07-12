"""Query normalization for memohood's recall path (DESIGN_v1.md: "gate (v1 =
pass-through) -> query_norm -> hybrid search via _engine.retrieve").

The core primitive is :func:`meaningful_terms` (spec name
``_meaningful_terms``, exported both ways below): strip Russian and English
stopwords/pronouns/question-words from a user query, but ALWAYS keep tokens
that look technical — CamelCase/PascalCase identifiers, ``UPPER_SNAKE``
constants, anything containing a digit (version strings, ``gpt-4``,
``2026.4.10``), and path-like tokens (``config.yaml``, ``C:/Users/...``,
``db.py``). Those are exactly the tokens a plain stopword-list scrub would
otherwise mangle or a naive "drop anything short" filter would throw away,
and they are disproportionately the tokens worth searching FTS/vectors for.

This mirrors the original ``_meaningful_terms`` gate ("нормализация запроса
`_meaningful_terms`" — one of the mechanisms explicitly named as a real,
verified-against-code mechanism worth keeping) but is a clean-room
reimplementation here, not a port: that original targeted English/generic
stopwords only; this project is RU-first ("у русского FTS нет морфологии...
проект RU-first"), so the RU stopword/pronoun/question-word list is the
larger and more load-bearing half.

Used by ``provider.py``'s ``prefetch()`` BEFORE handing the query to
``_engine.retrieve.hybrid_search``/``fts_search_messages`` — trims a
conversational question like "а что мы решили насчёт HERMES_HOME?" down to
the terms actually worth searching (``HERMES_HOME``), so the FTS leg isn't
diluted by function words that would otherwise both inflate the OR'd MATCH
expression and dominate the RU-stemmed side with near-universal stems.
"""

from __future__ import annotations

import re
from typing import List

# ---------------------------------------------------------------------------
# Stopword / pronoun / question-word lists
# ---------------------------------------------------------------------------

# Russian: personal/possessive/reflexive pronouns, question words, common
# conjunctions/particles/prepositions, and high-frequency auxiliary verb
# forms. Lowercase; matched against the lowercased token.
_RU_STOPWORDS = frozenset("""
я ты он она оно мы вы они меня тебя его её нас вас их мне тебе ему ей нам вам
им мной тобой им ей ими собой себя себе свой своя своё свои мой моя моё мои
твой твоя твоё твои наш наша наше наши ваш ваша ваше ваши
это этот эта это эти тот та то те такой такая такое такие
кто что какой какая какое какие который которая которое которые
где когда почему зачем отчего откуда куда сколько чей чья чьё чьи ли
и а но или либо да нет не ни же бы то и то так итак
в во на с со у к ко по для от до из изо о об обо при над под подо перед
про через без без сквозь между меж
что чтобы потому оттого поэтому также притом причём хотя если раз пока
уже ещё очень просто только лишь именно вот вон
есть быть был была было были будет будут являюсь являешься является
являемся являетесь являются
можно нужно надо нельзя
""".split())

# English: pronouns, articles, question words, common conjunctions/
# prepositions/auxiliaries.
_EN_STOPWORDS = frozenset("""
i you he she it we they me him her us them my your his its our their mine
yours hers ours theirs myself yourself himself herself itself ourselves
yourselves themselves
this that these those
what which who whom whose where when why how
a an the
and or but if then so because as
of to in on at by for with from about into onto over under above below
between among through during before after since until
than
do does did doing done
can could will would shall should may might must
not no yes
is are was were be been being am
""".split())

_STOPWORDS = _RU_STOPWORDS | _EN_STOPWORDS

# ---------------------------------------------------------------------------
# Tokenizer — keeps hyphens/dots/slashes/underscores INSIDE a token so
# "gpt-4", "2026.4.10", "HERMES_HOME", "C:/Users/name/config.yaml" survive
# as single tokens instead of being shredded at every punctuation mark.
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[^\s,;:!?()\[\]{}\"'«»]+", re.UNICODE)
_EDGE_STRIP_RE = re.compile(r"^[.\-_/\\]+|[.\-_/\\]+$")

_CAMEL_CASE_RE = re.compile(r"[a-zа-яё][A-ZА-ЯЁ]|[A-ZА-ЯЁ][a-zа-яё].*[A-ZА-ЯЁ]")
_UPPER_SNAKE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_HAS_DIGIT_RE = re.compile(r"\d")
_PATH_LIKE_RE = re.compile(r"[/\\]|\.[A-Za-z0-9]{1,5}$")


def _is_camel_case(tok: str) -> bool:
    """True for mixed-case identifiers like ``MemoryProvider``,
    ``getHermesHome``, ``sqlite_vec`` mixed with caps, etc. Deliberately
    excludes tokens that are ALL upper- or ALL lower-case (those are handled
    by the UPPER_SNAKE check and the ordinary stopword path respectively)."""
    if tok.isupper() or tok.islower():
        return False
    return bool(_CAMEL_CASE_RE.search(tok))


def _is_upper_snake(tok: str) -> bool:
    """True for ``HERMES_HOME``, ``GEMINI_API_KEY``, ``API`` (single-word
    all-caps also counts — short acronyms are technical terms, not
    stopwords)."""
    return bool(_UPPER_SNAKE_RE.match(tok)) and len(tok) >= 2


def _has_digit(tok: str) -> bool:
    return bool(_HAS_DIGIT_RE.search(tok))


def _is_path_like(tok: str) -> bool:
    return bool(_PATH_LIKE_RE.search(tok)) or tok.startswith("~")


def _is_keeper(tok: str) -> bool:
    """Any of the technical-token heuristics that override stopword
    removal — CamelCase, UPPER_SNAKE, contains a digit, or looks like a
    path/filename."""
    return _is_camel_case(tok) or _is_upper_snake(tok) or _has_digit(tok) or _is_path_like(tok)


def meaningful_terms(text: str) -> List[str]:
    """Return the meaningful terms of *text*, in order, deduplicated.

    Technical-looking tokens (CamelCase/PascalCase, ``UPPER_SNAKE``, any
    token containing a digit, or path/filename-shaped tokens) are ALWAYS
    kept, with their original casing preserved. Everything else is
    lowercased and dropped if it's a Russian or English stopword/pronoun/
    question-word, or shorter than 2 characters after stripping edge
    punctuation. Never raises; empty/falsy input returns ``[]``.
    """
    if not text:
        return []

    seen: set = set()
    out: List[str] = []
    for raw_tok in _TOKEN_RE.findall(text):
        tok = _EDGE_STRIP_RE.sub("", raw_tok)
        if not tok:
            continue

        if _is_keeper(tok):
            key = tok  # case-preserving identity for technical tokens
            if key not in seen:
                seen.add(key)
                out.append(tok)
            continue

        lower = tok.lower()
        if len(lower) < 2:
            continue
        if lower in _STOPWORDS:
            continue
        if lower not in seen:
            seen.add(lower)
            out.append(lower)

    return out


# Spec/private-name alias — DESIGN_v1.md and this project's internal design
# notes both refer to this function as ``_meaningful_terms``; keep that name
# importable too so code/tests written against the spec's naming work unchanged.
_meaningful_terms = meaningful_terms
