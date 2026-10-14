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
