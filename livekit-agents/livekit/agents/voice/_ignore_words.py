"""Shared ignore-word matching for interruption and late-acknowledgement handling."""

from __future__ import annotations

import string


def matches_ignore_words(text: str, ignore_words: list[str] | None) -> bool:
    """Return True if `text` can be composed entirely of ignore words,
    or is a partial (prefix) match where completed segments are ignore words
    and the trailing segment is a prefix of an ignore word."""
    if not text:
        return False

    text_cleaned = text.lower().strip().translate(str.maketrans("", "", string.punctuation))
    if not text_cleaned:
        return True

    text_cleaned = "".join(text_cleaned.split())
    if not text_cleaned:
        return True

    if not ignore_words:
        return False

    cleaned_ignore_words: list[str] = []
    for word in ignore_words:
        if not isinstance(word, str):
            continue
        cleaned = word.lower().strip().translate(str.maketrans("", "", string.punctuation))
        if cleaned:
            cleaned_ignore_words.append(cleaned)

    if not cleaned_ignore_words:
        return False

    if len(text_cleaned) == 1:
        return True

    dp = [False] * (len(text_cleaned) + 1)
    dp[0] = True
    for idx in range(1, len(text_cleaned) + 1):
        for word in cleaned_ignore_words:
            if (
                idx >= len(word)
                and dp[idx - len(word)]
                and text_cleaned[idx - len(word) : idx] == word
            ):
                dp[idx] = True
                break

    if dp[len(text_cleaned)]:
        return True

    for idx in range(len(text_cleaned)):
        if not dp[idx]:
            continue
        suffix = text_cleaned[idx:]
        for word in cleaned_ignore_words:
            if word.startswith(suffix) and word != suffix:
                return True

    return False


def _clean_for_exact_match(text: str) -> str:
    stripped = text.lower().translate(str.maketrans("", "", string.punctuation))
    return "".join(stripped.split())


def matches_ignore_words_exactly(text: str, ignore_words: list[str] | None) -> bool:
    """Return True only if the whole of `text` is made of ignore words.

    Unlike `matches_ignore_words`, which is tuned for interim transcripts, this accepts no
    single-character wildcard and no prefix of an ignore word, and empty text is not a match.
    Use it for final transcripts.
    """
    text_cleaned = _clean_for_exact_match(text)
    if not text_cleaned or not ignore_words:
        return False

    cleaned_ignore_words = [
        cleaned
        for word in ignore_words
        if isinstance(word, str) and (cleaned := _clean_for_exact_match(word))
    ]
    if not cleaned_ignore_words:
        return False

    reachable = [False] * (len(text_cleaned) + 1)
    reachable[0] = True
    for idx in range(1, len(text_cleaned) + 1):
        reachable[idx] = any(
            idx >= len(word) and reachable[idx - len(word)] and text_cleaned.endswith(word, 0, idx)
            for word in cleaned_ignore_words
        )
    return reachable[len(text_cleaned)]
