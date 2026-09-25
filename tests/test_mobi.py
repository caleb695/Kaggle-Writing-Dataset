"""MOBI reader tests.

The MOBI header field offsets caused a subtle, silent bug during development:
reading them relative to the ``MOBI`` magic instead of from the start of record 0
shifted every field by 16 bytes, so ``traildata_flags`` picked up unrelated
header bytes. The consequence was not a crash -- it was 1,160 mojibake
characters and ~9 kB of drift, with the text silently truncated mid-sentence.

The regression test for that is :func:`test_byte_stream_matches_declared_length`,
which is the strongest check available: the decompressed, trailing-stripped byte
stream must equal the length the file itself declares.
"""

from __future__ import annotations

import struct

import pytest

from writing_dataset.mobi import (
    MobiError,
    decompress_palmdoc,
    parse_header,
    strip_trailing_data,
)

# --------------------------------------------------------------------------- #
# PalmDOC decompression
# --------------------------------------------------------------------------- #


def test_palmdoc_literals_and_runs():
    assert decompress_palmdoc(b"Hello") == b"Hello"
    # 0x01-0x08 is a literal run of that many following bytes.
    assert decompress_palmdoc(b"\x03abc") == b"abc"
    # 0x00 is a literal NUL.
    assert decompress_palmdoc(b"\x00") == b"\x00"
    # 0xC0-0xFF emits a space plus (byte ^ 0x80).
    assert decompress_palmdoc(b"\xc1") == b" A"
    assert decompress_palmdoc(b"\xe1") == b" a"


def test_palmdoc_backreference_copies_overlapping_runs():
    """Distance 1 must repeat a single byte, not read past the start."""
    # Literal run of 2 ("ab"), then a reference: distance 1, length 3.
    pair = (1 << 3) | (3 - 3)
    out = decompress_palmdoc(b"\x02ab" + bytes([0x80 | (pair >> 8), pair & 0xFF]))
    assert out == b"abbbb"


def test_palmdoc_backreference_distance_and_length():
    # "XY" then copy distance 2, length 4 -> "XYXY XY" (length 4).
    pair = (2 << 3) | (4 - 3)
    out = decompress_palmdoc(b"\x02XY" + bytes([0x80 | (pair >> 8), pair & 0xFF]))
    assert out == b"XYXYXY"


def test_palmdoc_empty():
    assert decompress_palmdoc(b"") == b""


# --------------------------------------------------------------------------- #
# Trailing data
# --------------------------------------------------------------------------- #


def test_trailing_data_untouched_when_flags_zero():
    assert strip_trailing_data(b"hello world", 0) == b"hello world"


def test_trailing_data_multibyte_entry_length_from_last_byte():
    """With bit 0 set, the final byte encodes the entry length as (b & 3) + 1."""
    record = b"payload" + b"\xaa\xbb\x02"     # 0x02 & 3 == 2 -> strip 3
    assert strip_trailing_data(record, 1) == b"payload"


def test_trailing_data_variable_entry_with_high_bit_terminator():
    """With bit 1 set, a back-to-front varint gives the entry size."""
    assert strip_trailing_data(b"payload\x81", 2) == b"payload"


def test_trailing_data_only_strips_from_the_end():
    """Prose in the middle of a record must never be touched."""
    record = b"payload\x02"
    assert strip_trailing_data(record, 3).startswith(b"payload")


def test_trailing_data_never_strips_the_whole_record():
    """A pathological flag must not destroy the record."""
    record = b"tiny"
    out = strip_trailing_data(record, 0xFFFF)
    assert out == record


# --------------------------------------------------------------------------- #
# Header parsing
# --------------------------------------------------------------------------- #


def make_minimal_mobi(compression=2, encryption=0, version=6, traildata=3) -> bytes:
    """Build the smallest file with a parseable PDB + PalmDOC + MOBI header."""
    palm = struct.pack(">HHIHHH", compression, 0, 5, 1, 4096, encryption) + b"\x00\x00"

    mobi_len = 0x108
    mobi = bytearray(b"\x00" * mobi_len)
    mobi[0:4] = b"MOBI"
    struct.pack_into(">I", mobi, 0x04, mobi_len)
    struct.pack_into(">I", mobi, 0x0C, 65001)          # codepage
    struct.pack_into(">I", mobi, 0x14, version)        # file version
    struct.pack_into(">I", mobi, 0x58, 5)              # min_version (gate needs >= 5)
    struct.pack_into(">I", mobi, 0x98, 0xFFFFFFFF)     # drm_offset, absolute 0xA8
    struct.pack_into(">H", mobi, 0xE2, traildata)      # traildata (0xF2 absolute)

    rec0 = palm + bytes(mobi)
    text = b"\x03abc"
    records = [rec0, text]

    header = bytearray(78 + 8 * len(records))
    header[60:68] = b"BOOKMOBI"
    struct.pack_into(">H", header, 76, len(records))
    offset = len(header)
    for i, rec in enumerate(records):
        struct.pack_into(">I", header, 78 + 8 * i, offset)
        offset += len(rec)
    return bytes(header) + b"".join(records)


def test_parse_minimal_mobi(tmp_path):
    path = tmp_path / "min.mobi"
    path.write_bytes(make_minimal_mobi())

    header = parse_header(path)
    assert header.compression == 2
    assert header.encryption == 0
    assert header.mobi_version == 6
    assert header.text_encoding == 65001
    assert header.encoding == "utf-8"
    assert header.num_records == 2


def test_traildata_flags_read_from_absolute_offset(tmp_path):
    """Regression: the flags live at absolute 0xF2, not magic-relative 0xF2."""
    path = tmp_path / "min.mobi"
    path.write_bytes(make_minimal_mobi(traildata=0x0003))
    assert parse_header(path).traildata_flags == 0x0003

    path2 = tmp_path / "min2.mobi"
    path2.write_bytes(make_minimal_mobi(traildata=0x0000))
    assert parse_header(path2).traildata_flags == 0x0000


def test_unset_drm_offset_is_not_treated_as_drm(tmp_path):
    """0xFFFFFFFF is the 'unset' sentinel, not a record index."""
    path = tmp_path / "min.mobi"
    path.write_bytes(make_minimal_mobi())
    assert parse_header(path).drm_offset == 0xFFFFFFFF


def test_truncated_file_raises_mobierror(tmp_path):
    path = tmp_path / "tiny.mobi"
    path.write_bytes(b"\x00" * 10)
    with pytest.raises(MobiError):
        parse_header(path)


# --------------------------------------------------------------------------- #
# Real-file checks (skipped when the sources are not present)
# --------------------------------------------------------------------------- #

SAMPLES = [
    "data/raw/greek and roman mythology/Percy Jackson.mobi",
    "data/raw/greek and roman mythology/The Heroes of Olympus - The Complete Series.mobi",
]


def _existing():
    from pathlib import Path

    return [p for p in SAMPLES if Path(p).is_file()]


@pytest.mark.skipif(not _existing(), reason="source .mobi files not present")
@pytest.mark.parametrize("path", _existing())
def test_byte_stream_matches_declared_length(path):
    """The decompressed, stripped stream must equal the declared text length.

    This is the end-to-end proof that header offsets, trailing-data handling and
    decompression are all correct. Drift of even a few bytes means records are
    being spliced wrongly.
    """
    from pathlib import Path

    from writing_dataset.mobi import read_raw_text

    header = parse_header(Path(path))
    raw = read_raw_text(Path(path))
    assert len(raw) == header.text_length


@pytest.mark.skipif(not _existing(), reason="source .mobi files not present")
@pytest.mark.parametrize("path", _existing())
def test_no_mojibake_and_wellformed_ending(path):
    """The decoded text must be clean and end at the closing html tag."""
    from pathlib import Path

    from writing_dataset.mobi import read_html_text

    html, header = read_html_text(Path(path))
    assert html.count("\ufffd") == 0, "UTF-8 decoding produced replacement characters"
    assert html.rstrip().endswith("</html>")
    assert header.exth.get(100), "author should be present in EXTH metadata"


@pytest.mark.skipif(not _existing(), reason="source .mobi files not present")
@pytest.mark.parametrize("path", _existing())
def test_omnibus_splits_into_five_novels(path):
    from pathlib import Path

    from writing_dataset.mobi import novel_starts

    starts = novel_starts(Path(path))
    assert starts is not None
    assert len(starts) == 5, f"expected 5 novels, got {len(starts)}"
    labels = [s.label.lower() for s in starts]
    assert all(labels), "every novel start needs a title"
    # Every novel start is a real, strictly increasing byte offset.
    offsets = [s.offset for s in starts]
    assert offsets == sorted(offsets)
    assert len(set(offsets)) == len(offsets)


@pytest.mark.skipif(not _existing(), reason="source .mobi files not present")
def test_omnibus_segments_cover_the_whole_file():
    from pathlib import Path

    from writing_dataset.mobi import novel_starts, read_raw_text, segment_by_offsets

    path = Path(SAMPLES[0])
    raw = read_raw_text(path)
    starts = novel_starts(path, raw)
    segments = segment_by_offsets(raw, starts)
    assert sum(len(html.encode("utf-8")) for _, html in segments) == len(raw), (
        "segments must tile the file with no gaps and no overlap"
    )
