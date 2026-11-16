"""Moonshine's tokenizer, ported: token ids -> text, the way Moonshine's runtime makes it (BinTokenizer in
core/bin-tokenizer/bin-tokenizer.cpp, then Transcriber::sanitize_text in core/transcriber.cpp). Step 7 ports this to
Swift."""

import codecs
from pathlib import Path

from common.config import TOKENIZER_PATH, VOCABULARY_SIZE

WORD_START_MARKER = "▁"  # ▁: SentencePiece's mark for "a space goes before this piece"
TRIMMED_CHARACTERS = " \t"  # what Moonshine's trim() removes from both ends
INVALID_BYTE_REPLACEMENT_ERROR_HANDLER = "moonshine_invalid_byte_as_question_mark"


def _replace_one_invalid_byte_with_question_mark(error: UnicodeDecodeError) -> tuple[str, int]:
    """sanitize_text's rule: each byte that doesn't start a valid UTF-8 sequence becomes "?", and reading goes on at
    the next byte. Python's decoder is stricter than sanitize_text about a few sequences sanitize_text only checks
    the shape of (overlong forms, surrogates, past U+10FFFF); those become "?" here and pass through in Moonshine."""
    return "?", error.start + 1


codecs.register_error(INVALID_BYTE_REPLACEMENT_ERROR_HANDLER, _replace_one_invalid_byte_with_question_mark)


def read_tokenizer_entries(tokenizer_bytes: bytes) -> list[bytes]:
    """tokenizer.bin: one entry per token id, in id order. Each is a length, then that many bytes. The length is one
    byte if under 128; otherwise two, with length = second x 128 + first - 128. A length of 0 is an empty entry."""
    entries = []
    offset = 0
    while offset < len(tokenizer_bytes):
        first_length_byte = tokenizer_bytes[offset]
        offset += 1
        if first_length_byte < 128:
            byte_count = first_length_byte
        else:
            byte_count = tokenizer_bytes[offset] * 128 + first_length_byte - 128
            offset += 1
        if offset + byte_count > len(tokenizer_bytes):
            raise ValueError(f"tokenizer entry {len(entries)} runs past the end of the file")
        entries.append(tokenizer_bytes[offset:offset + byte_count])
        offset += byte_count
    return entries


def is_special_entry(entry: bytes) -> bool:
    """Moonshine skips these when making text: more than 2 bytes, starting with < and ending with >. In this file
    that's ids 0-2 (<unk>, <s>, </s>) and 32000-32767 (<<ST_0>> ... <<ST_767>>)."""
    return len(entry) > 2 and entry.startswith(b"<") and entry.endswith(b">")


class MoonshineTokenizer:
    """tokenizer.bin, read once. Ids 3-258 are single raw bytes (byte fallback: a character the vocabulary lacks is
    spelled as its UTF-8 bytes, one token each), so text is made by joining every token's bytes first and decoding
    UTF-8 after, never token by token."""

    def __init__(self, tokenizer_path: Path = TOKENIZER_PATH):
        self.entries_by_token_id = read_tokenizer_entries(tokenizer_path.read_bytes())
        if len(self.entries_by_token_id) != VOCABULARY_SIZE:
            raise ValueError(f"{tokenizer_path}: {len(self.entries_by_token_id)} entries, expected {VOCABULARY_SIZE}")

    def tokens_to_text(self, token_ids: list[int]) -> str:
        """Moonshine's text for these tokens: special entries skipped, the rest's bytes joined, invalid UTF-8 bytes
        as "?", ▁ as a space, spaces and tabs trimmed from both ends. An empty entry is an error, as in Moonshine
        ("Invalid token")."""
        text_bytes = bytearray()
        for token_id in token_ids:
            if not 0 <= token_id < len(self.entries_by_token_id):
                raise ValueError(f"token {token_id} is outside the vocabulary")
            entry = self.entries_by_token_id[token_id]
            if not entry:
                raise ValueError(f"invalid token {token_id}: its tokenizer entry is empty")
            if not is_special_entry(entry):
                text_bytes += entry
        text = text_bytes.decode("utf-8", errors=INVALID_BYTE_REPLACEMENT_ERROR_HANDLER)
        return text.replace(WORD_START_MARKER, " ").strip(TRIMMED_CHARACTERS)
