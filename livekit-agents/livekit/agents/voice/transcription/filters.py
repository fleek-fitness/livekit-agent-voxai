import re
import time
from collections.abc import AsyncIterable, Sequence
from typing import Literal

from ...log import logger

TextTransforms = Literal["filter_markdown", "filter_emoji"]


def apply_text_transforms(
    text: AsyncIterable[str], transforms: Sequence[TextTransforms]
) -> AsyncIterable[str]:
    all_transforms = {
        "filter_markdown": filter_markdown,
        "filter_emoji": filter_emoji,
    }

    for transform in transforms:
        if transform not in all_transforms:
            raise ValueError(
                f"Invalid transform: {transform}, available transforms: {all_transforms.keys()}"
            )
        text = all_transforms[transform](text)
    return text


LINE_PATTERNS = [
    # headers: remove # and following spaces
    (re.compile(r"^#{1,6}\s+", re.MULTILINE), ""),
    # list markers: remove -, +, * and following spaces
    (re.compile(r"^\s*[-+*]\s+", re.MULTILINE), ""),
    # block quotes: remove > and following spaces
    (re.compile(r"^\s*>\s+", re.MULTILINE), ""),
]

INLINE_PATTERNS = [
    # images: keep alt text ![alt](url) -> alt
    (re.compile(r"!\[([^\]]*)\]\([^)]*\)"), r"\1"),
    # links: keep text part [text](url) -> text
    (re.compile(r"\[([^\]]*)\]\([^)]*\)"), r"\1"),
    # bold: remove asterisks from **text** (not preceded/followed by non-whitespace)
    (re.compile(r"(?<!\S)\*\*([^*]+?)\*\*(?!\S)"), r"\1"),
    # italic: remove asterisks from *text* (not preceded/followed by non-whitespace)
    (re.compile(r"(?<!\S)\*([^*]+?)\*(?!\S)"), r"\1"),
    # bold with underscores: remove underscores from __text__ (word boundaries)
    (re.compile(r"(?<!\w)__([^_]+?)__(?!\w)"), r"\1"),
    # italic with underscores: remove underscores from _text_ (word boundaries)
    (re.compile(r"(?<!\w)_([^_]+?)_(?!\w)"), r"\1"),
    # code blocks: remove ``` from ```text```
    (re.compile(r"`{3,4}[\S]*"), ""),
    # inline code: remove ` from `text`
    (re.compile(r"`([^`]+?)`"), r"\1"),
    # strikethrough: remove ~~text~~ (no spaces next to tildes)
    (re.compile(r"~~(?!\s)([^~]*?)(?<!\s)~~"), ""),
]
INLINE_SPLIT_TOKENS = " ,.?!;，。？！；"

COMPLETE_LINKS_PATTERN = re.compile(r"\[[^\]]*\]\([^)]*\)")  # links [text](url)
COMPLETE_IMAGES_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]*\)")  # images ![text](url)


async def filter_markdown(text: AsyncIterable[str]) -> AsyncIterable[str]:
    """
    Filter out markdown symbols from the text.
    """

    last_emit_time = time.perf_counter()
    def has_incomplete_pattern(buffer: str) -> bool:
        """Check if buffer might contain incomplete markdown patterns that need more text."""

        if buffer.endswith(("#", "-", "+", "*", ">", "!", "`", "~", " ")):
            return True

        # check for incomplete bold (**text** or *text*)
        double_asterisks = buffer.count("**")
        if double_asterisks % 2 == 1:
            return True

        single_asterisks = buffer.count("*") - (double_asterisks * 2)
        if single_asterisks % 2 == 1:
            return True

        # check for incomplete underscores (__text__ or _text_)
        double_underscores = buffer.count("__")
        if double_underscores % 2 == 1:
            return True
        single_underscores = buffer.count("_") - (double_underscores * 2)
        if single_underscores % 2 == 1:
            return True

        # check for incomplete code (`text`)
        backticks = buffer.count("`")
        if backticks % 2 == 1:
            return True

        # check for incomplete strikethrough (~~text~~)
        double_tildes = buffer.count("~~")
        if double_tildes % 2 == 1:
            return True

        # check for incomplete links [text](url) or images ![text](url)
        open_brackets = buffer.count("[")
        complete_links = len(COMPLETE_LINKS_PATTERN.findall(buffer))
        complete_images = len(COMPLETE_IMAGES_PATTERN.findall(buffer))

        remaining_brackets = open_brackets - complete_links - complete_images
        if remaining_brackets > 0:
            return True

        return False

    def process_complete_text(text: str, is_newline: bool = False) -> str:
        if is_newline:
            for pattern, replacement in LINE_PATTERNS:
                text = pattern.sub(replacement, text)

        for pattern, replacement in INLINE_PATTERNS:
            text = pattern.sub(replacement, text)

        return text

    buffer = ""
    buffer_is_newline = True  # track if buffer is at start of line

    async for chunk in text:
        buffer += chunk

        if "\n" in buffer:
            lines = buffer.split("\n")
            buffer = lines[-1]  # keep last incomplete line

            for i, line in enumerate(lines[:-1]):
                is_newline = buffer_is_newline if i == 0 else True
                processed_line = process_complete_text(line, is_newline=is_newline)
                emit_delay = time.perf_counter() - last_emit_time
                if emit_delay > 0.05:
                    logger.debug(
                        f"md_filter_emit delay_ms={emit_delay*1000:.1f} buffer_len={len(line)} preview='{line[:40]}'"
                    )
                last_emit_time = time.perf_counter()
                yield processed_line + "\n"

            buffer_is_newline = True
            continue

        # split at the position after the split token
        last_split_pos = 0
        for token in INLINE_SPLIT_TOKENS:
            last_split_pos = max(last_split_pos, buffer.rfind(token, last_split_pos))
            if last_split_pos >= len(buffer) - 1:
                break

        if last_split_pos >= 1:
            processable = buffer[:last_split_pos]  # exclude the split token
            rest = buffer[last_split_pos:]
            if not has_incomplete_pattern(processable):
                emit_delay = time.perf_counter() - last_emit_time
                if emit_delay > 0.05:
                    logger.debug(
                        f"md_filter_emit delay_ms={emit_delay*1000:.1f} buffer_len={len(processable)} preview='{processable[:40]}'"
                    )
                last_emit_time = time.perf_counter()
                yield process_complete_text(processable, is_newline=buffer_is_newline)
                buffer = rest
                buffer_is_newline = False

    if buffer:
        emit_delay = time.perf_counter() - last_emit_time
        if emit_delay > 0.05:
            logger.debug(
                f"md_filter_emit_final delay_ms={emit_delay*1000:.1f} buffer_len={len(buffer)} preview='{buffer[:40]}'"
            )
        yield process_complete_text(buffer, is_newline=buffer_is_newline)


# Unicode block ranges from: https://unicode.org/Public/UNIDATA/Blocks.txt
EMOJI_PATTERN = re.compile(
    r"[\U0001F000-\U0001FBFF]"  # Emoji blocks: Mahjong Tiles through Symbols for Legacy Computing
    r"|[\U00002600-\U000026FF]"  # Miscellaneous Symbols
    r"|[\U00002700-\U000027BF]"  # Dingbats
    r"|[\U00002B00-\U00002BFF]"  # Miscellaneous Symbols and Arrows
    r"|[\U0000FE00-\U0000FE0F]"  # Variation selectors
    r"|\U0000200D"  # Zero width joiner
    r"|\U000020E3"  # Combining enclosing keycap
    r"+",
    re.UNICODE,
)


async def filter_emoji(text: AsyncIterable[str]) -> AsyncIterable[str]:
    """
    Filter out emojis from the text.
    """

    async for chunk in text:
        filtered_chunk = EMOJI_PATTERN.sub("", chunk)
        yield filtered_chunk
