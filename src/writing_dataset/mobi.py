"""Native MOBI / PalmDOC reader.

Why this exists instead of just using the ``mobi`` package
---------------------------------------------------------
``mobi`` (KindleUnpack) is a good general-purpose unpacker, but it crashed on
one of the source files::

    File ".../mobi_html.py", line 95, in insertHREFS
        imageName = rscnames[imageNumber - 1]
    IndexError: list index out of range

That happens when a ``<img recindex="N">`` points past the end of the resource
list, which is common in older conversions. Unpacking images is also wasted work
for a text corpus.

So this module reads the Palm Database header and the MOBI header directly and
returns only the HTML text stream. That covers what a prose corpus needs, works
on files KindleUnpack rejects, and keeps the dependency surface small.

Format reference (all multi-byte fields big-endian):

* PDB: 78-byte header; ``numRecords`` at 0x4C; then ``numRecords`` 8-byte
  record entries (4-byte file offset + 1 attribute byte + 3-byte unique id).
* Record 0: 16-byte PalmDOC header, then the MOBI header (``MOBI`` magic at
  record offset 16), then optionally an EXTH block.
* PalmDOC header: ``compression`` (2), unused (2), ``textLength`` (4),
  ``recordCount`` (2), ``recordSize`` (2), ``encryption`` (2), unused (2).
* Text: PDB records 1..``recordCount``, each independently compressed,
  concatenated and then truncated to ``textLength``.

Supported: ``compression`` 1 (none) and 2 (PalmDOC LZ77), ``encryption`` 0.
Huff/CDIC (17480) and KF8/AZW3 are handed to the ``mobi`` package when it is
installed, and otherwise reported clearly rather than silently mis-parsed.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

#: PalmDOC compression identifiers.
COMPRESSION_NONE = 1
COMPRESSION_PALMDOC = 2
COMPRESSION_HUFFCDIC = 17480

#: Text encoding identifiers found in the MOBI header.
TEXT_ENCODINGS = {
    1252: "cp1252",
    65001: "utf-8",
    65002: "utf-16",
}

#: MOBI file version at which the KF8 container starts.
KF8_VERSION = 8

#: MOBI header field offsets, measured from the START OF RECORD 0.
#: These are absolute, not relative to the "MOBI" magic at offset 0x10 --
#: getting that wrong shifts every read by 16 bytes and makes
#: ``traildata_flags`` pick up unrelated header bytes, which silently corrupts
#: the text stream. Values cross-checked against KindleUnpack's
#: ``mobi6_header`` / ``mobi8_header`` tables.
OFF_TEXT_LENGTH = 0x04
OFF_TEXT_RECORDS = 0x08
OFF_RECORD_SIZE = 0x0A
OFF_ENCRYPTION = 0x0C
OFF_MAGIC = 0x10
OFF_MOBI_LENGTH = 0x14
OFF_CODEPAGE = 0x1C
OFF_FILE_VERSION = 0x24
OFF_TITLE_OFFSET = 0x54
OFF_TITLE_LENGTH = 0x58
OFF_MIN_VERSION = 0x68
OFF_FIRST_RESOURCE = 0x6C
OFF_HUFF_OFFSET = 0x70
OFF_HUFF_COUNT = 0x74
OFF_DRM_OFFSET = 0xA8
OFF_TRAILDATA_FLAGS = 0xF2


class MobiError(RuntimeError):
    """Raised when a MOBI file cannot be read as plain text."""


# --------------------------------------------------------------------------- #
# Header parsing
# --------------------------------------------------------------------------- #


@dataclass
class MobiHeader:
    """The fields this reader actually needs, plus a little metadata."""

    num_records: int
    record_offsets: list[int]
    compression: int
    text_length: int
    record_count: int
    record_size: int
    encryption: int

    mobi_version: int = 0
    mobi_header_length: int = 0
    text_encoding: int = 65001
    first_image_index: int = 0
    traildata_flags: int = 0

    full_name: Optional[str] = None
    exth: dict[int, str] = None  # type: ignore[assignment]

    #: DRM fields; a non-zero offset means the text is encrypted.
    drm_offset: int = 0

    @property
    def encoding(self) -> str:
        return TEXT_ENCODINGS.get(self.text_encoding, "utf-8")

    @property
    def is_kf8(self) -> bool:
        return self.mobi_version >= KF8_VERSION


def _parse_pdb(data: bytes) -> tuple[int, list[int]]:
    if len(data) < 78:
        raise MobiError("file is too small to be a Palm Database")
    num_records = struct.unpack_from(">H", data, 76)[0]
    if num_records <= 0:
        raise MobiError("Palm Database declares zero records")

    offsets: list[int] = []
    for i in range(num_records):
        pos = 78 + i * 8
        if pos + 4 > len(data):
            raise MobiError("record info list is truncated")
        offsets.append(struct.unpack_from(">I", data, pos)[0])
    return num_records, offsets


def parse_header(path: Path) -> MobiHeader:
    """Read the PDB + PalmDOC + MOBI headers without decompressing any text."""
    data = Path(path).read_bytes()
    num_records, offsets = _parse_pdb(data)

    # Bounds for record 0.
    start = offsets[0]
    end = offsets[1] if len(offsets) > 1 else len(data)
    rec0 = data[start:end]
    if len(rec0) < 16:
        raise MobiError("record 0 is too small to hold a PalmDOC header")

    compression, _unused, text_length, record_count, record_size, encryption = struct.unpack_from(
        ">HHIHHH", rec0, 0
    )

    header = MobiHeader(
        num_records=num_records,
        record_offsets=offsets,
        compression=compression,
        text_length=text_length,
        record_count=record_count,
        record_size=record_size,
        encryption=encryption,
    )

    if rec0[OFF_MAGIC:OFF_MAGIC + 4] != b"MOBI":
        # Plain PalmDOC / older format: no MOBI header, so no metadata. The text
        # records are still readable, which is all that matters here.
        return header

    mobi_length = struct.unpack_from(">I", rec0, OFF_MOBI_LENGTH)[0]
    header.mobi_header_length = mobi_length
    header.text_encoding = struct.unpack_from(">I", rec0, OFF_CODEPAGE)[0]
    header.mobi_version = struct.unpack_from(">I", rec0, OFF_FILE_VERSION)[0]

    def field(offset: int, fmt: str = ">I", default: int = 0) -> int:
        """Read a header field, tolerating headers shorter than the field."""
        if offset + struct.calcsize(fmt) > OFF_MAGIC + mobi_length:
            return default
        if offset + struct.calcsize(fmt) > len(rec0):
            return default
        return struct.unpack_from(fmt, rec0, offset)[0]

    full_name_offset = field(OFF_TITLE_OFFSET)
    full_name_length = field(OFF_TITLE_LENGTH)
    header.first_image_index = field(OFF_FIRST_RESOURCE, ">I", num_records)
    header.drm_offset = field(OFF_DRM_OFFSET)

    # The trailing-data flag is only meaningful for headers new enough to have
    # the field; older files carry no trailing data at all.
    min_version = field(OFF_MIN_VERSION, ">I", header.mobi_version)
    if mobi_length >= 0xE4 and min_version >= 5:
        header.traildata_flags = field(OFF_TRAILDATA_FLAGS, ">H", 0)

    if full_name_length and 0 < full_name_length < 4096:
        blob = rec0[full_name_offset:full_name_offset + full_name_length]
        header.full_name = blob.decode(header.encoding, errors="replace").strip("\x00").strip()

    # EXTH rides immediately after the MOBI header.
    exth_start = OFF_MAGIC + mobi_length
    header.exth = _parse_exth(rec0, exth_start, header.encoding)

    return header


def _parse_exth(rec0: bytes, offset: int, encoding: str) -> dict[int, str]:
    """Parse the EXTH metadata block into ``{type: value}``.

    Values that are not UTF-8 text are skipped rather than mis-decoded: several
    EXTH types are binary (cover offsets, KF8 boundaries).
    """
    out: dict[int, str] = {}
    if rec0[offset:offset + 4] != b"EXTH":
        return out

    exth_length, count = struct.unpack_from(">II", rec0, offset + 4)
    limit = min(len(rec0), offset + exth_length)
    pos = offset + 12

    for _ in range(count):
        if pos + 8 > limit:
            break
        rec_type, rec_length = struct.unpack_from(">II", rec0, pos)
        if rec_length < 8 or pos + rec_length > limit:
            break
        raw = rec0[pos + 8:pos + rec_length]
        try:
            value = raw.decode(encoding)
            # Reject control characters: those fields are binary, not text.
            if not any(ord(c) < 32 and c not in "\t\n\r" for c in value):
                out[rec_type] = value.strip("\x00").strip()
        except (UnicodeDecodeError, LookupError):
            pass
        pos += rec_length
    return out


# --------------------------------------------------------------------------- #
# PalmDOC decompression
# --------------------------------------------------------------------------- #


def decompress_palmdoc(data: bytes) -> bytes:
    """Decompress one PalmDOC (LZ77) record.

    Opcode table::

        0x00        literal NUL
        0x01-0x08   copy the next N bytes verbatim
        0x09-0x7F   literal byte
        0x80-0xBF   two-byte back-reference: distance = (pair >> 3) & 0x7FF,
                    length = (pair & 7) + 3
        0xC0-0xFF   a space followed by (byte ^ 0x80)

    The back-reference copies byte-by-byte so overlapping runs (distance 1)
    repeat correctly.
    """
    out = bytearray()
    i = 0
    n = len(data)

    while i < n:
        c = data[i]
        i += 1

        if 1 <= c <= 8:
            out += data[i:i + c]
            i += c
        elif c < 0x80:
            out.append(c)
        elif c >= 0xC0:
            out.append(0x20)
            out.append(c ^ 0x80)
        else:
            if i >= n:
                break
            pair = (c << 8) | data[i]
            i += 1
            distance = (pair >> 3) & 0x07FF
            length = (pair & 0x07) + 3
            if distance == 0:
                # Degenerate reference; emitting NULs matches the other readers.
                out += b"\x00" * length
            else:
                for _ in range(length):
                    out.append(out[-distance])

    return bytes(out)


def _trailing_data_size(data: bytes, size: int) -> int:
    """Length of one variable-width trailing entry, read back-to-front."""
    bitpos, result = 0, 0
    while True:
        if size <= 0:
            return result
        v = data[size - 1]
        result |= (v & 0x7F) << bitpos
        bitpos += 7
        size -= 1
        if (v & 0x80) != 0 or bitpos >= 28 or size == 0:
            return result


def strip_trailing_data(record: bytes, flags: int) -> bytes:
    """Remove the non-text bytes MOBI appends to the end of a record.

    Only called when ``traildata_flags`` is non-zero; with flags 0 (the usual
    case for the sources here) nothing is stripped.
    """
    if not flags or not record:
        return record

    count = 0
    test = flags >> 1
    while test:
        if test & 1:
            count += _trailing_data_size(record, len(record) - count)
        test >>= 1
    if flags & 1:
        idx = len(record) - count - 1
        if idx >= 0:
            count += (record[idx] & 0x3) + 1

    if count <= 0 or count >= len(record):
        return record
    return record[:-count]


# --------------------------------------------------------------------------- #
# Text extraction
# --------------------------------------------------------------------------- #


def read_html_text(path: Path) -> tuple[str, MobiHeader]:
    """Return ``(html_text, header)`` for a MOBI file.

    Raises :class:`MobiError` with an actionable message when the file needs a
    path this reader does not implement (DRM, Huff/CDIC, KF8).
    """
    header = parse_header(path)

    # 0xFFFFFFFF is the "unset" sentinel used throughout the MOBI header, not a
    # real record index, so only a plausible in-range offset means actual DRM.
    drm_present = header.encryption not in (0, None) or (
        0 < header.drm_offset < header.num_records
    )
    if drm_present:
        raise MobiError(
            f"{Path(path).name}: text is DRM-encrypted (encryption={header.encryption}, "
            f"drm_offset={header.drm_offset}). Remove DRM in Calibre first."
        )

    if header.is_kf8:
        text = _read_kf8_text(path)
        if text is not None:
            return text, header
        raise MobiError(
            f"{Path(path).name}: MOBI version {header.mobi_version} is KF8/AZW3, whose text "
            "lives in a different container. Install the `mobi` package "
            "(`pip install mobi`) to read it, or convert to EPUB in Calibre."
        )

    if header.compression == COMPRESSION_HUFFCDIC:
        text = _read_huffcdic_text(path, header)
        if text is not None:
            return text, header
        raise MobiError(
            f"{Path(path).name}: Huff/CDIC compression (17480) needs the `mobi` package. "
            "Install it (`pip install mobi`) or convert to EPUB in Calibre."
        )

    if header.compression not in (COMPRESSION_NONE, COMPRESSION_PALMDOC):
        raise MobiError(
            f"{Path(path).name}: unsupported compression {header.compression}; "
            "expected 1 (none) or 2 (PalmDOC)."
        )

    data = Path(path).read_bytes()
    pieces: list[bytes] = []
    for i in range(1, header.record_count + 1):
        if i + 1 > len(header.record_offsets):
            break
        start = header.record_offsets[i]
        end = (
            header.record_offsets[i + 1]
            if i + 1 < len(header.record_offsets)
            else len(data)
        )
        record = data[start:end]
        if header.traildata_flags:
            record = strip_trailing_data(record, header.traildata_flags)
        if header.compression == COMPRESSION_PALMDOC:
            record = decompress_palmdoc(record)
        pieces.append(record)

    raw = b"".join(pieces)
    # The stream is continuous across records, so trailing bytes from the last
    # record are cut by the declared text length.
    if header.text_length and 0 < header.text_length <= len(raw):
        raw = raw[:header.text_length]

    return raw.decode(header.encoding, errors="replace"), header


# --------------------------------------------------------------------------- #
# Delegated paths
# --------------------------------------------------------------------------- #


def _read_kf8_text(path: Path) -> Optional[str]:
    """Extract HTML from KF8/AZW3 via the ``mobi`` package, if installed."""
    html = _kindleunpack_html(path)
    if html is None:
        return None
    return html


def _read_huffcdic_text(path: Path, header: MobiHeader) -> Optional[str]:
    """Decompress Huff/CDIC text via the ``mobi`` package, if installed."""
    try:
        from mobi.mobi_uncompress import HuffcdicReader
    except Exception:
        return None

    try:
        data = Path(path).read_bytes()
        offsets = header.record_offsets

        def record(i: int) -> bytes:
            start = offsets[i]
            end = offsets[i + 1] if i + 1 < len(offsets) else len(data)
            return data[start:end]

        huff_offset = struct.unpack_from(">I", record(0), OFF_HUFF_OFFSET)[0]
        huff_count = struct.unpack_from(">I", record(0), OFF_HUFF_COUNT)[0]
        if not huff_count:
            return None

        reader = HuffcdicReader()
        reader.loadHuff(record(huff_offset))
        for i in range(1, huff_count):
            reader.loadCdic(record(huff_offset + i))

        pieces = [
            reader.unpack(record(i)) for i in range(1, header.record_count + 1)
            if i < len(offsets)
        ]
        raw = b"".join(pieces)
        if header.text_length:
            raw = raw[:header.text_length]
        return raw.decode(header.encoding, errors="replace")
    except Exception:
        return None


def read_raw_text(path: Path) -> bytes:
    """Return the decompressed, undecompressed text stream as raw bytes.

    Needed because ``<a filepos=N>`` anchors index this byte stream, not the
    decoded string: Python indexes ``str`` by character, so using the decoded
    text for offsets introduces drift that grows with position.
    """
    data = Path(path).read_bytes()
    header = parse_header(path)
    pieces: list[bytes] = []

    for i in range(1, header.record_count + 1):
        if i + 1 > len(header.record_offsets):
            break
        start = header.record_offsets[i]
        end = (
            header.record_offsets[i + 1]
            if i + 1 < len(header.record_offsets)
            else len(data)
        )
        record = data[start:end]
        if header.traildata_flags:
            record = strip_trailing_data(record, header.traildata_flags)
        if header.compression == COMPRESSION_PALMDOC:
            record = decompress_palmdoc(record)
        pieces.append(record)

    raw = b"".join(pieces)
    if header.text_length and 0 < header.text_length <= len(raw):
        raw = raw[:header.text_length]
    return raw


# --------------------------------------------------------------------------- #
# Table of contents  (used for omnibus splitting)
# --------------------------------------------------------------------------- #


@dataclass
class TocEntry:
    """One ``<a filepos=N>`` target from the book's own table of contents."""

    offset: int
    label: str

    def __repr__(self) -> str:  # pragma: no cover
        return f"TocEntry({self.offset}, {self.label!r})"


def _clean_label(raw: str) -> str:
    import regex

    text = regex.sub(r"<[^>]*>", " ", raw)
    text = (
        text.replace("&#160;", " ")
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&#8217;", "\u2019")
        .replace("&#8216;", "\u2018")
        .replace("&#8220;", "\u201c")
        .replace("&#8221;", "\u201d")
    )
    text = regex.sub(r"\s+", " ", text).strip()
    # Labels are cut out of a tag stream, so they often inherit a stray ">".
    return text.lstrip("> ").strip()


def toc_entries(path: Path, raw: Optional[bytes] = None) -> list[TocEntry]:
    """Extract ``filepos`` anchors and their labels, deduplicated and sorted.

    Labels may sit either before or after the ``<a>`` tag depending on how the
    source was converted, so both layouts are probed.
    """
    import regex

    if raw is None:
        raw = read_raw_text(path)
    text = raw.decode("utf-8", errors="replace")

    entries: dict[int, TocEntry] = {}
    for m in regex.finditer(r"<a\b[^>]*?filepos\s*=\s*0*(\d+)", text):
        offset = int(m.group(1))
        if offset >= len(raw):
            continue

        tail = text[m.end():m.end() + 400]
        label = _clean_label(regex.split(r"</a>", tail)[0])[:120]

        if not label:
            head = text[max(0, m.start() - 400):m.start()]
            label = _clean_label(head.split(">")[-1])[:120]

        if offset in entries:
            if not entries[offset].label and label:
                entries[offset] = TocEntry(offset, label)
            continue
        entries[offset] = TocEntry(offset, label)

    return sorted(entries.values(), key=lambda e: e.offset)


def _plain_text_at(raw: bytes, offset: int, length: int = 300) -> str:
    import regex

    chunk = raw[offset:offset + length].decode("utf-8", errors="replace")
    return regex.sub(r"\s+", " ", regex.sub(r"<[^>]*>", " ", chunk)).strip()


def novel_starts(
    path: Path,
    raw: Optional[bytes] = None,
    min_cluster_bytes: int = 50_000,
    contents_window: int = 6_000,
) -> Optional[list[TocEntry]]:
    """Locate the start of every novel inside a multi-book file.

    A candidate start is a table-of-contents entry whose target opens a fresh
    front-matter block, which is visible as a "Contents" page within the next
    few kilobytes. Glossaries and indexes produce dense false positives, so
    candidates are clustered: entries closer together than
    ``min_cluster_bytes`` belong to the same block, and only the last of each
    cluster is kept. That last entry is the novel title itself, because a novel
    begins with its title page and then its own contents list.

    Returns the ordered novel starts, or ``None`` when the file is a single
    book. Each start is a real offset in the raw byte stream.
    """
    if raw is None:
        raw = read_raw_text(path)
    entries = toc_entries(path, raw)
    if len(entries) < 2:
        return None

    candidates = [
        e
        for e in entries
        if "Contents" in raw[e.offset:e.offset + contents_window].decode("utf-8", "replace")
        or "Table of Contents"
        in raw[e.offset:e.offset + contents_window].decode("utf-8", "replace")
    ]
    if len(candidates) < 2:
        return None

    clusters: list[list[TocEntry]] = [[candidates[0]]]
    for entry in candidates[1:]:
        if entry.offset - clusters[-1][-1].offset < min_cluster_bytes:
            clusters[-1].append(entry)
        else:
            clusters.append([entry])

    starts: list[TocEntry] = []
    for cluster in clusters:
        pick = cluster[-1]
        if not pick.label:
            pick = TocEntry(pick.offset, _plain_text_at(raw, pick.offset, 60))
        starts.append(pick)

    return starts if len(starts) >= 2 else None


def segment_by_offsets(
    raw: bytes,
    starts: Sequence[TocEntry],
    encoding: str = "utf-8",
) -> list[tuple[str, str]]:
    """Slice the raw stream into ``[(title, html_text), ...]``, one per novel.

    The first segment always begins at byte 0 rather than at ``starts[0].offset``.
    Those leading bytes are the collection's own front matter (cover, imprint,
    collection contents), and starting at the first detected offset silently
    discarded them -- about 7.6 kB on the sample files. Folding them into the
    first novel means the segments tile the file exactly, with no gaps and no
    overlap, which :func:`writing_dataset.mobi.segment_by_offsets` callers can
    rely on.
    """
    boundaries = [0]
    for entry in starts[1:]:
        if 0 < entry.offset < len(raw):
            boundaries.append(entry.offset)
    boundaries = sorted(set(boundaries))

    titles = [s.label for s in starts] + [""] * max(0, len(boundaries) - len(starts))

    segments: list[tuple[str, str]] = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(raw)
        if end <= start:
            continue
        segments.append(
            (titles[i] if i < len(titles) else "", raw[start:end].decode(encoding, errors="replace"))
        )
    return segments


def detect_omnibus(path: Path, **kwargs) -> Optional[list[TocEntry]]:
    """Backwards-compatible alias for :func:`novel_starts`."""
    return novel_starts(path, **kwargs)


def _html_from_epub_zip(path: Path) -> Optional[str]:
    """Concatenate spine-ish XHTML from an EPUB zip (KF8 unpack output)."""
    import zipfile

    try:
        with zipfile.ZipFile(path) as zf:
            names = [
                n for n in zf.namelist()
                if n.lower().endswith((".xhtml", ".html", ".htm", ".xml"))
                and "meta-inf" not in n.lower()
                and not n.lower().endswith((".opf", ".ncx"))
            ]
            names.sort()
            parts: list[str] = []
            for name in names:
                lower = name.lower()
                if any(h in lower for h in ("toc", "nav", "cover", "copyright")):
                    continue
                try:
                    data = zf.read(name)
                except Exception:
                    continue
                if data[:2] == b"PK":
                    continue
                text = data.decode("utf-8", errors="replace")
                if "<p" in text.lower() or "<html" in text.lower() or "<body" in text.lower():
                    parts.append(text)
            return "\n".join(parts) if parts else None
    except Exception:
        return None


def _kindleunpack_html(path: Path) -> Optional[str]:
    """Run KindleUnpack and return the produced HTML, if it succeeds.

    KF8 unpack writes an EPUB zip, not a ``.html`` file. Reading that zip as
    UTF-8 used to return the PK header as "text".
    """
    import shutil

    try:
        import mobi
    except Exception:
        return None

    tempdir = None
    try:
        tempdir, out = mobi.extract(str(path))
    except Exception:
        return None

    try:
        out_path = Path(out)
        if not out_path.is_file():
            return None
        head = out_path.read_bytes()[:4]
        if out_path.suffix.lower() == ".epub" or head == b"PK\x03\x04":
            return _html_from_epub_zip(out_path)
        try:
            return out_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
    finally:
        if tempdir:
            shutil.rmtree(tempdir, ignore_errors=True)
