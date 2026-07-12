"""Russian stemming for memohood's FTS5 BM25 legs (``captures_fts.content_stem``
and ``messages_fts``'s stemmed column).

STEMMER DECISION: **PyStemmer** (BSD-2-Clause; wraps the C Snowball library, which itself ships
a Russian stemming algorithm). ``PyStemmer==3.1.0`` publishes prebuilt wheels
for cp311 on both Windows (``win_amd64``) and Linux (``manylinux``) -- no
compiler needed on either the Windows dev box or a Linux VPS. Declared in
this plugin's own ``plugin.yaml`` `pip_dependencies` (lazy-installed by
hermes' memory-provider loader, ``hermes_cli/memory_setup.py``).

This module MUST import cleanly even before PyStemmer is installed -- the
``Stemmer`` import is deferred into the first call to :func:`stem_ru`, and
any failure degrades to a lowercase-tokenized passthrough with a one-time
warning, never an ``ImportError`` at module load time.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Optional, Union

logger = logging.getLogger("memohood.stem")

_lock = threading.Lock()
_stemmer: Optional[Union[object, bool]] = None  # None=not checked yet, False=unavailable
_unavailable_warned = False

# Letters only (Unicode-aware) — punctuation/digits are token boundaries and
# are dropped from the stemmed output. This side-channel (`content_stem`/
# messages_fts stem column) is used ONLY for FTS matching; captures and
# recalled messages are always shown to the user/verified against their raw,
# unstemmed text, so dropping punctuation here can never corrupt a citation
# or a recall snippet.
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _get_stemmer():
    global _stemmer, _unavailable_warned
    if _stemmer is not None:
        return _stemmer
    with _lock:
        if _stemmer is not None:
            return _stemmer
        try:
            import Stemmer  # PyStemmer

            _stemmer = Stemmer.Stemmer("russian")
        except ImportError:
            if not _unavailable_warned:
                logger.warning(
                    "PyStemmer not installed - stem_ru() will pass text through "
                    "lowercased/tokenized but UNSTEMMED (RU BM25 leg degraded to "
                    "exact-token match only). It is declared in plugin.yaml's "
                    "pip_dependencies for lazy-install via `hermes memory setup`."
                )
                _unavailable_warned = True
            _stemmer = False
    return _stemmer


def stem_ru(text: str) -> str:
    """Return *text* with each word replaced by its Russian Snowball stem,
    space-joined (e.g. ``"договора"`` and ``"договор"`` both stem to the
    same token, so FTS matches across inflected forms).

    Degrades to a lowercase, tokenized (but unstemmed) passthrough if
    PyStemmer is not installed or raises for any reason — this function
    never raises and never returns ``None``. Empty/falsy input returns
    ``""``.

    NOT for display — this is purely an index/query-time transform for the
    FTS5 stemmed shadow column; the raw capture/message text is what gets
    shown to the user.
    """
    if not text:
        return ""

    stemmer = _get_stemmer()
    words = _WORD_RE.findall(text.lower())
    if not words:
        return ""

    if stemmer is False:
        return " ".join(words)

    try:
        stemmed = stemmer.stemWords(words)
    except Exception:
        logger.debug("PyStemmer.stemWords failed; falling back to raw tokens", exc_info=True)
        return " ".join(words)

    return " ".join(stemmed)
