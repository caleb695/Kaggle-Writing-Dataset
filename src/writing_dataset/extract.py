"""Source-file extraction: ``.epub`` / ``.txt`` / ``.md`` / ``.docx`` / ``.html`` -> :class:`Block` list.

Every extractor produces the same flat, source-ordered ``list[Block]``. All
book structure inference happens afterwards in :mod:`writing_dataset.structure`,
so a new format only ever needs to implement "walk this document and tell me
what is a heading, what is a scene break, and what is a paragraph".
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

import regex

from .types import BLOCK_HEADING, BLOCK_PARAGRAPH, BLOCK_SCENE_BREAK, Block

# --------------------------------------------------------------------------- #
# Cleaning helpers
# --------------------------------------------------------------------------- #

#: Characters that never carry meaning in a book and always indicate encoding dirt.
_JUNK_CHARS = dict.fromkeys(
    map(ord, "\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad\u2060"), None
)

#: Runs of whitespace inside a paragraph collapse to a single space.
_WS_RUN = regex.compile(r"[^\S\n]+")
#: Three or more newlines collapse to two (paragraph separator).
_NL_RUN = regex.compile(r"\n{3,}")
#: Space that precedes terminal punctuation, a common EPUB artefact.
_SPACE_BEFORE_PUNCT = regex.compile(r"\s+([,.;:!?…])")


def clean_inline(text: str) -> str:
    """Normalise a single paragraph's text without touching its semantics."""
    if not text:
        return ""
    # NFC only -- NFKC would rewrite typographic quotes, em dashes and ligatures,
    # destroying authorial style that this dataset exists to capture.
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_JUNK_CHARS)
    text = text.replace("\u00a0", " ").replace("\u2028", " ").replace("\u2029", " ")
    text = _WS_RUN.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    return text.strip()


def clean_heading(text: str) -> str:
    text = clean_inline(text)
    text = text.strip(" .\u2014-\u2013")  # drop trailing chapter-number dotted leaders
    return text.strip()


def _normalise_newlines(text: str) -> str:
    return _NL_RUN.sub("\n\n", text.replace("\r\n", "\n").replace("\r", "\n"))


def dehyphenate(text: str) -> str:
    """Repair words broken across hard-wrapped lines (plain-text sources only).

    Only applies when the fragment after the hyphen starts lowercase, which is
    the normal typographic convention. ``well-known`` at a line end is left
    alone because the continuation is lowercase too -- we accept that small
    false-positive rate rather than lose real hyphenated compounds.
    """
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def strip_page_number_lines(lines: list[str]) -> list[str]:
    """Drop lines that are obviously running heads or page numbers.

    Opt-in: a stray ``12`` inside real prose is rare but a wrong deletion is
    unrecoverable, so this is never enabled by default.
    """
    out = []
    for ln in lines:
        s = ln.strip()
        if re.fullmatch(r"\d{1,4}", s):
            continue
        if re.fullmatch(r"(?i)page\s+\d{1,4}( of \d{1,4})?", s):
            continue
        out.append(ln)
    return out


# --------------------------------------------------------------------------- #
# Scene-break / heading recognition (shared with structure.py)
# --------------------------------------------------------------------------- #

#: Characters used to build typographic scene separators.
ORNAMENT_CHARS = "\u002a\u2022\u00b7\u2043\u2053\u2731\u2732\u2733\u2734\u2735" \
                 "\u2736\u2737\u2738\u2739\u273a\u273b\u273c\u273d\u274a\u274b" \
                 "\u274c\u274d\u274e\u2756\u2758\u2759\u275a\u275b\u275c\u275d" \
                 "\u275e\u2042\u2044\u2045\u2046\u2020\u2021\u00a7\u00b6\u2016" \
                 "\u2017\u2018\u2019\u201c\u201d\u204c\u204d\u204e\u204f\u2050" \
                 "\u2051\u2052\u2054\u2055\u2056\u2057\u2058\u2059\u205a\u205b" \
                 "\u205c\u205d\u205e\u2660\u2661\u2662\u2663\u2664\u2665\u2666" \
                 "\u2667\u2668\u2669\u266a\u266b\u266c\u266d\u266e\u266f\u25c6" \
                 "\u25c7\u25a0\u25a1\u25aa\u25ab\u2b1a\u2b1b\u2b1c\u2b1d\u2b1e" \
                 "\u2059\u273f\u2740\u2741\u2742\u2743\u2744\u2745\u2746\u2747" \
                 "\u2748\u2749\u2750\u2751\u2752\u2753\u2754\u2755\u25cf\u25cb" \
                 "\u25ce\u25c9\u25d0\u25d1\u25e6\u26ab\u26aa\u2b24\u2b25\u2b26" \
                 "\u2b27\u2b28\u2b29\u2b2a\u2b2b\u2b2c\u2b2d\u2b2e\u2b2f" \
                 "\u0023\u007e\u003d\u002d\u2013\u2014\u005f\u002e\u0021\u003f" \
                 "\u2049\u00a1\u00bf"

# NOTE: the character class is built from escaped characters. A bare "-" inside
# a class defines a range -- placing it as written between "=" (U+003D) and
# "-" (U+2013) would silently create the range U+003D-U+2013, which contains
# every letter and digit, and every short sentence would look like an ornament.
_ORNAMENT_CLASS = "".join(regex.escape(c) for c in sorted(set(ORNAMENT_CHARS)))
_ORNAMENT_RE = regex.compile(rf"^[{_ORNAMENT_CLASS}\s]{{3,40}}$")
#: Single-character ornaments that unambiguously mean "scene break".
_STRONG_SINGLE = set("\u2042\u2043\u2044\u2053\u2731\u2732\u2733\u2734\u2735"
                     "\u274a\u274b\u274c\u274d\u274e\u2756\u2758\u2759\u275a"
                     "\u275b\u275c\u275d\u275e\u2660\u2661\u2662\u2663\u2664"
                     "\u2665\u2666\u2667\u2668\u2669\u266a\u266b\u266c\u266d"
                     "\u266e\u266f\u25c6\u25c7\u25a0\u25a1\u25aa\u25ab\u2b1a"
                     "\u2020\u2021\u00a7\u00b6\u007e\u2049")

#: "Chapter 1", "CHAPTER TWELVE", "Part II", "Prologue", "Epilogue", "Book Three"
_CHAPTER_WORD = (
    r"(?:chapter|part|book|section|canto|volume|act|scene|stave|"
    r"prologue|epilogue|interlude|afterword|foreword|preface|"
    r"introduction|appendix|postscript|envoi|proem)"
)
_CHAPTER_NUM = r"(?:\d{1,3}|[ivxlcdm]{1,7}|one|two|three|four|five|six|seven|eight|nine|ten|" \
               r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|" \
               r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)"
_CHAPTER_RE = regex.compile(
    rf"^\s*(?:{_CHAPTER_WORD})\b[\s.:\u2014\u2013-]*({_CHAPTER_NUM})?\b[^\n]{{0,60}}$",
    regex.IGNORECASE,
)
_MD_HEADING_RE = regex.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")


def is_scene_break_line(line: str, aggressive: bool = False) -> bool:
    """Decide whether a standalone line is a scene separator.

    Conservative by default: a lone ``-``/``--``/``--`` line is *not* treated as
    a break, because in dialogue-heavy prose a bare dash line is ambiguous.
    Pass ``aggressive=True`` to accept those too.
    """
    s = line.strip()
    if not s or len(s) > 40:
        return False
    if s in _STRONG_SINGLE:
        return True
    if len(s) >= 3 and _ORNAMENT_RE.match(s):
        # Require at least 3 non-space ornament characters.
        if sum(1 for ch in s if not ch.isspace()) >= 3:
            return True
    if aggressive and s and len(set(s)) == 1 and s[0] in "\u002d\u2013\u2014\u005f\u003d\u002a":
        return True
    return False


def is_numbered_scene_break(line: str) -> bool:
    """``1``, ``2``, ``3`` alone on a line -- common in some indie layouts."""
    s = line.strip()
    return bool(re.fullmatch(r"[1-9]\d{0,2}", s))


# --------------------------------------------------------------------------- #
# Extraction result
# --------------------------------------------------------------------------- #


@dataclass
class ExtractedDoc:
    source_path: Path
    blocks: list[Block] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    @property
    def n_paragraphs(self) -> int:
        return sum(1 for b in self.blocks if b.kind == BLOCK_PARAGRAPH)

    @property
    def n_words(self) -> int:
        return sum(len(b.text.split()) for b in self.blocks if b.kind == BLOCK_PARAGRAPH)


# --------------------------------------------------------------------------- #
# Plain text / Markdown
# --------------------------------------------------------------------------- #

_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


def read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    for enc in _ENCODINGS:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def extract_text(
    path: Path,
    aggressive_breaks: bool = False,
    numbered_breaks: bool = False,
    strip_pages: bool = False,
) -> ExtractedDoc:
    """Parse a plain-text or Markdown file into blocks.

    A blank line ends a paragraph. Markdown ``#`` headings and ``Chapter N``
    style lines become headings; ornament-only lines become scene breaks.
    """
    text = _normalise_newlines(read_text_file(path))
    lines = text.split("\n")
    if strip_pages:
        lines = strip_page_number_lines(lines)

    blocks: list[Block] = []
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        raw = dehyphenate("\n".join(buffer))
        joined = " ".join(part.strip() for part in raw.split("\n") if part.strip())
        cleaned = clean_inline(joined)
        if cleaned:
            blocks.append(Block(BLOCK_PARAGRAPH, cleaned, origin=str(path.name)))
        buffer.clear()

    for ln in lines:
        stripped = ln.strip()

        if not stripped:  # blank line -> paragraph boundary
            flush()
            continue

        md = _MD_HEADING_RE.match(ln)
        if md:
            flush()
            level = 1 if len(md.group(1)) <= 2 else 2
            title = clean_heading(md.group(2))
            if title:
                blocks.append(Block(BLOCK_HEADING, title, level=level, origin=path.name))
            continue

        if is_scene_break_line(stripped, aggressive=aggressive_breaks):
            flush()
            blocks.append(Block(BLOCK_SCENE_BREAK, "", origin=path.name))
            continue

        if numbered_breaks and is_numbered_scene_break(stripped):
            flush()
            blocks.append(Block(BLOCK_SCENE_BREAK, "", origin=path.name))
            continue

        # A short standalone line that looks like a chapter heading.
        if len(stripped) <= 70 and _CHAPTER_RE.match(stripped) and not buffer:
            flush()
            blocks.append(
                Block(BLOCK_HEADING, clean_heading(stripped), level=1, origin=path.name)
            )
            continue

        buffer.append(ln)

    flush()
    return ExtractedDoc(path, blocks, {"title": path.stem})


# --------------------------------------------------------------------------- #
# HTML / XHTML (also the backbone of the EPUB reader)
# --------------------------------------------------------------------------- #

_BLOCK_TAGS = {
    "p", "div", "section", "article", "blockquote", "aside", "figure",
    "figcaption", "td", "li", "dd", "dt", "pre", "address",
}
_HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
_SKIP_TAGS = {
    "script", "style", "nav", "head", "title", "meta", "link",
    # Media and layout elements: they carry no prose. Tag names here are
    # compared after stripping any namespace prefix, so "mbp:pagebreak" -> "pagebreak".
    "img", "image", "svg", "figure", "audio", "video", "object", "iframe",
    "pagebreak", "pagetemplate", "pagemap", "pagemapproc",
}
_BREAK_HINT_CLASSES = (
    "scene-break", "scenebreak", "dinkus", "section-break", "sectionbreak",
    "separator", "divider", "ornament", "asterism", "hr", "break", "space",
)
_HEADING_HINT_CLASSES = ("chapter", "title", "heading", "heading1", "chapter-title")


def _with_br(html_text: str) -> str:
    """Mark ``<br>`` before parsing so malformed tags still split.

    Uses a sentinel rather than a newline: pretty-printed XHTML also contains
    newlines inside ``<p>`` (soft wraps), and those must not become paragraph
    boundaries.
    """
    return re.sub(r"(?i)<br\s*/?>", _BR_SENTINEL, html_text)


def _text_of(node) -> str:
    """Visible text of a node, with inline markup resolved away."""
    try:
        return clean_inline(node.get_text(" ").replace("\r", " ").replace("\n", " "))
    except Exception:  # pragma: no cover - defensive
        return ""


#: Private-use sentinel so ``<br>`` splits survive ``get_text`` joining.
_BR_SENTINEL = "\ue000"

#: Class tokens that mean "this paragraph is a chapter title", even without
#: an ``<h1>``. Common in older EPUB conversions (``ChapNo``, ``Chapter``).
_CHAPTER_CLASS_TOKENS = {
    "chapter", "chapno", "chap-no", "chaptitle", "chaptertitle",
    "chapter-title", "chapter-number", "chapternumber", "chapterhead",
    "chapter-head", "chaptername", "chapter-name", "chapternum",
}


def _leaf_pieces(node) -> list[str]:
    """Split a leaf block's text on ``<br>`` boundaries only.

    Soft line-wraps inside a ``<p>`` -- pretty-printed XHTML with a newline
    every ~60 characters -- are not paragraph boundaries and must be
    collapsed to spaces. Honouring them as splits turned one Fablehaven
    omnibus into 51k one-line "paragraphs". ``<br>`` is the only hard break.
    """
    try:
        for br in node.find_all("br"):
            br.replace_with(_BR_SENTINEL)
        raw = node.get_text(" ")
    except Exception:  # pragma: no cover - defensive
        return []
    pieces: list[str] = []
    for part in raw.split(_BR_SENTINEL):
        collapsed = clean_inline(part.replace("\r", " ").replace("\n", " "))
        if collapsed:
            pieces.append(collapsed)
    return pieces


def _class_id_blob(node) -> str:
    cls = node.get("class") or []
    if isinstance(cls, str):
        cls = cls.split()
    return " ".join(list(cls) + [node.get("id") or "", node.get("epub:type") or ""]).lower()


#: Words that are page furniture, never chapter titles.
FURNITURE_HEADINGS = {
    "contents", "table of contents", "cover", "title page", "copyright",
    "copyright page", "dedication", "acknowledgements", "acknowledgments",
    "about the author", "about the publisher", "also by", "praise",
    "advertisement", "colophon", "epigraph", "half title", "end matter",
    "front matter", "back matter", "glossary", "index", "notes",
}

#: Terminal punctuation that marks prose rather than a display line.
_SENTENCE_END = (".", "?", "!", "\u201d", '"')

_ALIGN_CENTER_VALUES = {"center", "centre", "middle"}
_PLACEHOLDER_ALIGN_VALUES = {"", "0", "0%", "0pt", "0px", "none", "justify", "left", "right"}


def _max_font_size(node) -> int:
    """Largest ``<font size>`` inside ``node`` as an integer weight (0 if none).

    Handles both absolute sizes (``size="6"``) and relative ones (``size="+2"``).
    """
    best = 0
    for font in node.find_all("font"):
        raw = (font.get("size") or "").strip()
        if not raw:
            continue
        try:
            if raw.startswith("+"):
                best = max(best, 3 + int(raw[1:]))
            elif raw.startswith("-"):
                continue
            else:
                best = max(best, int(raw))
        except ValueError:
            continue
    return best


def _is_centered(node) -> bool:
    align = (node.get("align") or "").strip().lower()
    if align in _ALIGN_CENTER_VALUES:
        return True
    # EPUB3 / CSS-driven centring.
    blob = _class_id_blob(node)
    return "center" in blob or "centre" in blob


def _display_heading_level(node, text: str) -> int:
    """Classify a short, styled block as a heading, or return 0.

    Many conversions mark chapter titles only by appearance: a centred line in a
    large font, with no semantic tag. The test is kept deliberately tight --
    short, few words, no sentence-ending punctuation, and either centred or in a
    large font -- so ordinary prose and verse lines are not promoted.

    Such a line is reported as chapter level (1), because in these conversions a
    centred display line *is* the chapter title. Reporting it as a mere
    sub-heading would leave a novel showing one chapter plus sixty scenes, which
    misrepresents the structure even though the chunk boundaries end up the
    same. The cost of being wrong is a small extra chapter for a dedication or
    centred epigraph, which is harmless: the text and its heading stay together
    either way.
    """
    if not text or len(text) > 60:
        return 0
    # A real heading has words. This rejects ornaments, rules and pure
    # punctuation that happen to be centred.
    if not any(ch.isalnum() for ch in text):
        return 0
    if text.endswith(_SENTENCE_END):
        return 0
    words = text.split()
    if not words or len(words) > 8:
        return 0

    size = _max_font_size(node)
    centered = _is_centered(node)

    if size >= 5:              # size="6" chapter title / POV name
        return 1
    if size >= 3 and centered:
        return 1
    if centered and len(words) <= 4 and len(text) <= 34:
        return 1
    return 0


def _has_block_child(node) -> bool:
    return any(
        getattr(c, "name", None) in (_BLOCK_TAGS | set(_HEADING_TAGS) | {"hr", "ul", "ol", "table"})
        for c in getattr(node, "children", [])
    )


def _iter_blocks(node, origin: str, doc_level: int) -> Iterator[Block]:
    """Depth-first walk emitting blocks in document order."""
    from bs4 import NavigableString

    for child in getattr(node, "children", []):
        name = getattr(child, "name", None)

        if name is None:  # NavigableString
            if isinstance(child, NavigableString):
                txt = clean_inline(str(child))
                # Bare text directly under <body>/<div>. Require real content so
                # whitespace between block elements does not become a paragraph.
                if len(txt) > 1 and any(ch.isalnum() for ch in txt):
                    if is_scene_break_line(txt):
                        yield Block(BLOCK_SCENE_BREAK, "", origin=origin)
                    else:
                        yield Block(BLOCK_PARAGRAPH, txt, origin=origin)
            continue

        lname = name.lower()
        if lname in _SKIP_TAGS:
            continue

        if lname == "hr":
            yield Block(BLOCK_SCENE_BREAK, "", origin=origin)
            continue

        if lname in _HEADING_TAGS:
            txt = clean_heading(_text_of(child))
            if txt:
                lvl = _HEADING_TAGS[lname]
                # Demote/promote so the document's own top heading is chapter level.
                level = doc_level if lvl <= 2 else 2
                yield Block(BLOCK_HEADING, txt, level=level, origin=origin)
            continue

        if lname in _BLOCK_TAGS:
            blob = _class_id_blob(child)

            # An empty div acting as an ornament.
            txt = _text_of(child)
            if not txt:
                if lname == "div" and any(h in blob for h in _BREAK_HINT_CLASSES):
                    yield Block(BLOCK_SCENE_BREAK, "", origin=origin)
                continue

            # A styled separator holding a dinkus.
            if any(h in blob for h in _BREAK_HINT_CLASSES) and len(txt) <= 40 \
                    and is_scene_break_line(txt):
                yield Block(BLOCK_SCENE_BREAK, "", origin=origin)
                continue

            if any(h in blob for h in _HEADING_HINT_CLASSES) and len(txt) <= 90 \
                    and lname == "p" and _CHAPTER_RE.match(txt):
                yield Block(BLOCK_HEADING, clean_heading(txt), level=1, origin=origin)
                continue

            # An ornament is a scene break, never a heading. This has to be
            # checked before the appearance test, because a centred "* * *"
            # satisfies every "short centred line" condition.
            if is_scene_break_line(txt):
                yield Block(BLOCK_SCENE_BREAK, "", origin=origin)
                continue

            # Class-tagged chapter titles (``<p class="Chapter">``, ``ChapNo``)
            # are headings even when they don't match "Chapter N".
            classes = child.get("class") or []
            if isinstance(classes, str):
                classes = classes.split()
            class_tokens = {c.lower() for c in classes}
            if (
                lname == "p"
                and class_tokens & _CHAPTER_CLASS_TOKENS
                and txt
                and len(txt) <= 90
                and any(ch.isalnum() for ch in txt)
            ):
                yield Block(BLOCK_HEADING, clean_heading(txt), level=1, origin=origin)
                continue

            # Appearance-based heading: a short centred / large-font line with
            # no semantic tag (typical of MOBI conversions).
            if not _has_block_child(child):
                level = _display_heading_level(child, txt)
                if level:
                    yield Block(BLOCK_HEADING, clean_heading(txt), level=level,
                                origin=origin)
                    continue

            if _has_block_child(child):
                yield from _iter_blocks(child, origin, doc_level)
                continue

            # Leaf block whose content may itself be multi-line via <br>.
            for piece in _leaf_pieces(child):
                if not piece:
                    continue
                if is_scene_break_line(piece):
                    yield Block(BLOCK_SCENE_BREAK, "", origin=origin)
                else:
                    yield Block(BLOCK_PARAGRAPH, piece, origin=origin)
            continue

        # Inline element at block level (e.g. bare <span>/<em> text).
        if _has_block_child(child):
            yield from _iter_blocks(child, origin, doc_level)
        else:
            txt = _text_of(child)
            if txt:
                yield Block(BLOCK_PARAGRAPH, txt, origin=origin)


def html_to_blocks(
    html_text: str,
    origin: str,
    doc_level: int = 1,
    xml: Optional[bool] = None,
) -> list[Block]:
    """Parse an HTML/XHTML string into blocks.

    ``xml`` selects the parser: ``True`` for the strict XML parser, ``False``
    for the lenient HTML one, ``None`` to auto-detect. MOBI content forces the
    lenient parser, because PalmDOC markup is frequently not well-formed XML
    (unclosed ``<img>`` and ``<p>`` tags are normal).
    """
    from bs4 import BeautifulSoup

    if xml is None:
        xml = origin.endswith((".xhtml", ".xml")) or "<?xml" in html_text[:200]

    soup = BeautifulSoup(_with_br(html_text), "lxml-xml" if xml else "lxml")
    for tag in soup.find_all(lambda t: _local(t) in _SKIP_TAGS):
        tag.decompose()
    root = soup.body or soup
    return [
        b for b in _iter_blocks(root, origin, doc_level)
        if b.kind != BLOCK_PARAGRAPH or len(b.text.split()) > 0
    ]


def extract_html(path: Path) -> ExtractedDoc:
    text = read_text_file(path)
    return ExtractedDoc(path, html_to_blocks(text, path.name), {"title": path.stem})


# --------------------------------------------------------------------------- #
# EPUB
# --------------------------------------------------------------------------- #

#: Name fragments that identify navigational or furniture documents.
_SKIP_EPUB_HINTS = (
    "nav", "toc", "contents", "cover", "copyright", "titlepage", "title-page",
    "colophon", "advert", "also-by", "newsletter", "acknowledg",
)

_HTML_MEDIA_TYPES = {
    "application/xhtml+xml", "text/html", "application/x-dtbook+xml", "",
}


def _local(tag) -> str:
    """Local name of a tag, ignoring any namespace prefix."""
    return (getattr(tag, "name", "") or "").split(":")[-1]


def _find_all_local(soup, name: str):
    return soup.find_all(lambda t: _local(t) == name)


def _read_epub_documents(path: Path):
    """Return ``(metadata, [(name, html_text), ...])`` in spine order.

    Self-contained on purpose: ``ebooklib`` raises ``lxml.etree.ParserError`` on
    real-world EPUBs that contain an empty nav document, which is common. The
    container -> OPF -> spine path is small enough to do directly and degrades
    gracefully when any single step fails.
    """
    import posixpath
    import zipfile
    from urllib.parse import unquote

    from bs4 import BeautifulSoup

    meta: dict = {}
    documents: list[tuple[str, str]] = []

    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        lower_map = {n.lower(): n for n in names}

        def read(name: str) -> Optional[bytes]:
            actual = lower_map.get(name.lower())
            if actual is None:
                return None
            try:
                return zf.read(actual)
            except Exception:
                return None

        opf_path: Optional[str] = None
        container = read("META-INF/container.xml")
        if container:
            try:
                soup = BeautifulSoup(container, "lxml-xml")
                rootfile = soup.find("rootfile")
                if rootfile is not None:
                    opf_path = rootfile.get("full-path")
            except Exception:
                opf_path = None
        if not opf_path:
            opf_path = next((n for n in names if n.lower().endswith(".opf")), None)

        manifest_items: list[tuple[str, str, str]] = []  # (name, media_type, properties)
        ordered: list[tuple[str, str, str]] = []

        if opf_path:
            raw = read(opf_path)
            if raw:
                try:
                    opf = BeautifulSoup(raw, "lxml-xml")
                    base = posixpath.dirname(opf_path)

                    for tag in _find_all_local(opf, "title"):
                        if tag.get_text(strip=True):
                            meta["title"] = clean_heading(tag.get_text(strip=True))
                            break
                    for tag in _find_all_local(opf, "creator"):
                        if tag.get_text(strip=True):
                            meta["creator"] = tag.get_text(strip=True)
                            break
                    for tag in _find_all_local(opf, "description"):
                        blob = tag.get_text(" ", strip=True)
                        if blob:
                            meta["description"] = clean_inline(blob)[:2000]
                            break

                    by_id: dict[str, tuple[str, str, str]] = {}
                    for item in _find_all_local(opf, "item"):
                        href = item.get("href") or ""
                        media = (item.get("media-type") or "").strip().lower()
                        props = (item.get("properties") or "").strip()
                        if not href:
                            continue
                        resolved = posixpath.normpath(
                            posixpath.join(base, unquote(href)) if base else unquote(href)
                        )
                        entry = (resolved, media, props)
                        manifest_items.append(entry)
                        if item.get("id"):
                            by_id[item.get("id")] = entry

                    for ref in _find_all_local(opf, "itemref"):
                        entry = by_id.get(ref.get("idref"))
                        if entry is not None:
                            ordered.append(entry)
                except Exception:
                    pass

        def looks_like_html(name: str, media: str, props: str) -> bool:
            if "nav" in props.lower():
                return False
            if media in _HTML_MEDIA_TYPES and name.lower().endswith(
                (".xhtml", ".html", ".htm", ".xml")
            ):
                return True
            return name.lower().endswith((".xhtml", ".html", ".htm"))

        candidates = [
            e for e in (ordered or manifest_items) if looks_like_html(*e)
        ]
        if not candidates:  # last resort: anything HTML-looking in the archive
            candidates = [
                (n, "application/xhtml+xml", "")
                for n in names
                if n.lower().endswith((".xhtml", ".html", ".htm"))
            ]

        seen: set[str] = set()
        for name, _media, _props in candidates:
            if name in seen:
                continue
            seen.add(name)
            raw = read(name)
            if not raw:
                continue
            documents.append((name, raw.decode("utf-8", errors="replace")))

    return meta, documents


def _epub_documents_to_blocks(documents: list[tuple[str, str]]) -> list[Block]:
    blocks: list[Block] = []
    for name, content in documents:
        blob = name.lower()
        if any(h in blob for h in _SKIP_EPUB_HINTS):
            continue
        blocks.extend(html_to_blocks(content, f"epub:{name}"))
    return blocks


def extract_epub(path: Path) -> ExtractedDoc:
    """Parse an EPUB in spine order into a single document.

    Multi-novel collections are split by :func:`extract_epub_collection` (used
    by :func:`extract_file`); this helper stays one-doc so existing tests and
    callers that want the raw concatenation still work.
    """
    meta, documents = _read_epub_documents(path)
    blocks = _epub_documents_to_blocks(documents)
    title = meta.get("title") or path.stem
    return ExtractedDoc(path, blocks, {"title": title, **meta})


# --------------------------------------------------------------------------- #
# EPUB omnibus splitting
# --------------------------------------------------------------------------- #

_OMNIBUS_HINT = regex.compile(
    r"\b(trilogy|omnibus|boxed[\s-]?set|complete series|complete collection|"
    r"e-?book boxed|boxed e-?book)\b",
    regex.IGNORECASE,
)
_BOOK_NAV_RE = regex.compile(
    r"^\s*(?:book|volume|tome)\s+"
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"i{1,3}|iv|v|vi{0,3}|ix|\d{1,2})\b",
    regex.IGNORECASE,
)
_ISBN13_RE = regex.compile(r"(97[89]\d{10})")
_FURNITURE_NAV_LABELS = {
    "cover", "copyright", "contents", "table of contents", "title page",
    "dedication", "acknowledgements", "acknowledgments", "about the author",
    "about", "also by", "colophon", "title",
}


def _is_furniture_nav_label(label: str) -> bool:
    s = (label or "").strip().lower()
    if not s or s in _FURNITURE_NAV_LABELS:
        return True
    if s.startswith("about "):
        return True
    return False


def _read_epub_ncx(path: Path) -> tuple[Optional[str], list[tuple[str, str, int, str]]]:
    """Return ``(docTitle, [(label, src, n_children, first_child_label), ...])`` from the NCX.

    Only *top-level* navPoints are returned. Nested chapter entries are
    counted so we can tell a novel (many children) from a furniture leaf.
    """
    import zipfile
    from bs4 import BeautifulSoup

    try:
        with zipfile.ZipFile(path) as zf:
            ncx_name = next((n for n in zf.namelist() if n.lower().endswith(".ncx")), None)
            if not ncx_name:
                return None, []
            soup = BeautifulSoup(zf.read(ncx_name), "lxml-xml")
    except Exception:
        return None, []

    doc_title = None
    for tag in soup.find_all(True):
        if _local(tag).lower() == "doctitle":
            text = tag.get_text(" ", strip=True)
            if text:
                doc_title = clean_heading(text)
                break

    navmap = None
    for tag in soup.find_all(True):
        if _local(tag).lower() == "navmap":
            navmap = tag
            break
    if navmap is None:
        return doc_title, []

    roots: list[tuple[str, str, int, str]] = []
    for child in getattr(navmap, "children", []):
        if _local(child).lower() != "navpoint":
            continue
        label = ""
        src = ""
        n_kids = 0
        first_child_label = ""
        for sub in child.find_all(True, recursive=False):
            lname = _local(sub).lower()
            if lname == "navlabel" and not label:
                label = sub.get_text(" ", strip=True)
            elif lname == "content" and not src:
                src = (sub.get("src") or "").split("#")[0]
            elif lname == "navpoint":
                n_kids += 1
                if not first_child_label:
                    nested_label = sub.find("text")
                    if nested_label is not None:
                        first_child_label = nested_label.get_text(" ", strip=True)
        if not label:
            text_tag = child.find("text")
            if text_tag is not None:
                label = text_tag.get_text(" ", strip=True)
        if not src:
            content_tag = child.find("content")
            if content_tag is not None:
                src = (content_tag.get("src") or "").split("#")[0]
        roots.append((label, src, n_kids, first_child_label))
    return doc_title, roots


def _doc_index_for_src(documents: list[tuple[str, str]], src: str) -> Optional[int]:
    """Locate a spine document for an NCX ``src``.

    Basename-only matching is unsafe: a collection root ``titlepage.xhtml``
    would steal ``3/titlepage.xhtml`` (Inheritance Cycle: Brisingr vanished
    into Eldest). Prefer an exact or directory-qualified match; fall back to
    basename only when ``src`` itself has no directory component.
    """
    import posixpath

    if not src:
        return None
    src = src.split("#")[0].lstrip("./")
    for i, (name, _) in enumerate(documents):
        if name == src or name.endswith("/" + src):
            return i
    if "/" not in src.replace("\\", "/"):
        base = posixpath.basename(src)
        for i, (name, _) in enumerate(documents):
            if posixpath.basename(name) == base:
                return i
    return None


def _isbn_novel_groups(
    documents: list[tuple[str, str]],
    ncx_roots: list[tuple[str, str, int, str]],
) -> Optional[list[tuple[str, list[tuple[str, str]]]]]:
    """Cluster spine documents by a 13-digit ISBN in the filename.

    Boxed-set EPUBs (Beyonders) ship three novels whose files are named
    ``9781416997986_ch01.html``, ``9781416997993_ch01.html``, ...
    """
    groups: dict[str, list[tuple[str, str]]] = {}
    order: list[str] = []
    for item in documents:
        m = _ISBN13_RE.search(item[0])
        if not m:
            continue
        isbn = m.group(1)
        if isbn not in groups:
            groups[isbn] = []
            order.append(isbn)
        groups[isbn].append(item)
    if len(order) < 2:
        return None
    if any(len(groups[i]) < 3 for i in order):
        return None

    # Prefer the NCX entry with the most nested children so an "About the
    # author" leaf that happens to live under the last book's ISBN does not
    # overwrite the novel title (Beyonders: Chasing the Prophecy vs author bio).
    best: dict[str, tuple[int, str]] = {}
    for label, src, n_kids, _first in ncx_roots:
        m = _ISBN13_RE.search(src or "")
        if not m or _is_furniture_nav_label(label):
            continue
        isbn = m.group(1)
        prev = best.get(isbn)
        if prev is None or n_kids > prev[0]:
            best[isbn] = (n_kids, clean_heading(label))

    return [(best[i][1] if i in best else i, groups[i]) for i in order]


def _ncx_novel_groups(
    documents: list[tuple[str, str]],
    ncx_roots: list[tuple[str, str, int, str]],
    *,
    require_book_label: bool,
) -> Optional[list[tuple[str, list[tuple[str, str]]]]]:
    """Slice the spine at top-level NCX entries that look like novels."""
    candidates: list[tuple[str, str, int, str]] = []
    for label, src, n_kids, first_child in ncx_roots:
        if n_kids < 5:
            continue
        if _is_furniture_nav_label(label):
            continue
        if require_book_label and not _BOOK_NAV_RE.match(label or ""):
            continue
        candidates.append((label, src, n_kids, first_child))
    if len(candidates) < 2:
        return None

    starts: list[tuple[str, int]] = []
    for label, src, _n, first_child in candidates:
        idx = _doc_index_for_src(documents, src)
        if idx is None:
            continue
        # "BOOK ONE" / first nested "Fablehaven" -> use the nested title.
        title = label
        if _BOOK_NAV_RE.match(label or "") and first_child and not _BOOK_NAV_RE.match(first_child):
            title = first_child
        starts.append((clean_heading(title) or f"Book {len(starts) + 1}", idx))
    if len(starts) < 2:
        return None

    starts.sort(key=lambda x: x[1])
    # Drop duplicate indexes (two labels pointing at the same file).
    dedup: list[tuple[str, int]] = []
    seen: set[int] = set()
    for title, idx in starts:
        if idx in seen:
            continue
        seen.add(idx)
        dedup.append((title, idx))
    if len(dedup) < 2:
        return None

    groups: list[tuple[str, list[tuple[str, str]]]] = []
    for i, (title, idx) in enumerate(dedup):
        start = 0 if i == 0 else idx
        end = dedup[i + 1][1] if i + 1 < len(dedup) else len(documents)
        if end <= start:
            continue
        groups.append((title, documents[start:end]))
    return groups if len(groups) >= 2 else None


def _epub_novel_groups(
    path: Path,
    meta: dict,
    documents: list[tuple[str, str]],
) -> Optional[list[tuple[str, list[tuple[str, str]]]]]:
    """Detect a multi-novel EPUB and return ``[(title, documents), ...]``.

    Conservative on purpose: a false split would shatter a single novel into
    several "books". We only cut when at least one of these is true:

    * spine filenames cluster by two or more ISBN-13 prefixes;
    * the NCX has two or more top-level ``BOOK ONE`` / ``VOLUME 2`` entries;
    * the collection title/description says trilogy/omnibus/complete series
      *and* the NCX has two or more nested novel-sized roots.
    """
    ncx_title, ncx_roots = _read_epub_ncx(path)
    blob = " ".join(
        s for s in (ncx_title, meta.get("title"), meta.get("description")) if s
    )
    omnibus_hint = bool(_OMNIBUS_HINT.search(blob))

    isbn_groups = _isbn_novel_groups(documents, ncx_roots)
    if isbn_groups:
        return isbn_groups

    labeled = _ncx_novel_groups(documents, ncx_roots, require_book_label=True)
    if labeled:
        return labeled

    if omnibus_hint:
        nested = _ncx_novel_groups(documents, ncx_roots, require_book_label=False)
        if nested:
            return nested
    return None


def extract_epub_collection(path: Path, split_omnibus: bool = True) -> list[ExtractedDoc]:
    """Parse an EPUB, splitting a boxed-set/omnibus into one document per novel."""
    meta, documents = _read_epub_documents(path)
    collection = meta.get("title") or path.stem

    groups = _epub_novel_groups(path, meta, documents) if split_omnibus else None
    if not groups:
        blocks = _epub_documents_to_blocks(documents)
        return [ExtractedDoc(path, blocks, {"title": collection, **meta})]

    docs: list[ExtractedDoc] = []
    for i, (title, docs_slice) in enumerate(groups):
        blocks = _epub_documents_to_blocks(docs_slice)
        docs.append(
            ExtractedDoc(
                path,
                blocks,
                {
                    **meta,
                    "title": title or f"{collection} — book {i + 1}",
                    "collection": collection,
                    "part_index": i + 1,
                    "part_count": len(groups),
                },
            )
        )
    return docs


# --------------------------------------------------------------------------- #
# MOBI / AZW  (Kindle)
# --------------------------------------------------------------------------- #


def _extract_kf8(path: Path, header, split_omnibus: bool = True) -> Optional[list[ExtractedDoc]]:
    """KF8/AZW3 text lives in an EPUB container. Unpack, then reuse the EPUB path.

    KindleUnpack's ``mobi.extract`` returns the path of that EPUB zip, not
    HTML. Reading it as UTF-8 produced two garbage "paragraphs" for
    ``The False Prince.azw3``.
    """
    import shutil

    tempdir = None
    try:
        import mobi as mobi_pkg
    except Exception:
        return None
    try:
        tempdir, out = mobi_pkg.extract(str(path))
        out_path = Path(out)
        if not out_path.is_file():
            return None
        head = out_path.read_bytes()[:4]
        if out_path.suffix.lower() == ".epub" or head == b"PK\x03\x04":
            docs = extract_epub_collection(out_path, split_omnibus=split_omnibus)
            collection = header.full_name or (header.exth or {}).get(503) or path.stem
            for doc in docs:
                doc.source_path = path
                meta = dict(doc.metadata or {})
                if header.exth:
                    meta.setdefault("author", header.exth.get(100))
                    meta.setdefault("publisher", header.exth.get(101))
                meta["mobi_version"] = header.mobi_version
                meta.setdefault("collection", collection)
                if not meta.get("title"):
                    meta["title"] = collection
                doc.metadata = {k: v for k, v in meta.items() if v is not None}
            return docs
        return None
    except Exception:
        return None
    finally:
        if tempdir:
            shutil.rmtree(tempdir, ignore_errors=True)


def extract_mobi(path: Path, split_omnibus: bool = True) -> list[ExtractedDoc]:
    """Parse a Kindle file into one or more documents.

    Returns a list because a "Complete Series" omnibus contains several novels
    in one file. The training rules forbid joining text from different books, so
    a multi-novel file is split at the table-of-contents offsets the file itself
    declares, and each novel becomes its own book downstream.

    PalmDOC markup is frequently malformed XML, so the lenient HTML parser is
    forced. ``<img>`` and ``<mbp:pagebreak>`` carry no prose and are skipped by
    :func:`html_to_blocks`.
    """
    from .mobi import (
        MobiError,
        novel_starts,
        parse_header,
        read_html_text,
        read_raw_text,
        segment_by_offsets,
    )

    try:
        header = parse_header(path)
    except MobiError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise MobiError(f"{path.name}: could not read MOBI data ({type(exc).__name__}: {exc})")

    if header.is_kf8:
        docs = _extract_kf8(path, header, split_omnibus=split_omnibus)
        if docs:
            return docs
        raise MobiError(
            f"{path.name}: MOBI version {header.mobi_version} is KF8/AZW3, whose text "
            "lives in a different container. Install the `mobi` package "
            "(`pip install mobi`) to read it, or convert to EPUB in Calibre."
        )

    try:
        html_text, header = read_html_text(path)
        raw = read_raw_text(path)
    except MobiError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise MobiError(f"{path.name}: could not read MOBI data ({type(exc).__name__}: {exc})")

    base_meta = {
        "author": (header.exth or {}).get(100),
        "publisher": (header.exth or {}).get(101),
        "language": (header.exth or {}).get(524),
        "isbn": (header.exth or {}).get(104),
        "mobi_version": header.mobi_version,
        "compression": header.compression,
    }
    collection = header.full_name or (header.exth or {}).get(503) or path.stem

    starts = novel_starts(path, raw) if split_omnibus else None

    if not starts:
        blocks = html_to_blocks(html_text, f"mobi:{path.name}", xml=False)
        meta = {k: v for k, v in {**base_meta, "title": collection}.items() if v is not None}
        return [ExtractedDoc(path, blocks, meta)]

    docs: list[ExtractedDoc] = []
    for i, (title, segment_html) in enumerate(segment_by_offsets(raw, starts, header.encoding)):
        label = (title or "").strip() or f"{collection} — book {i + 1}"
        blocks = html_to_blocks(segment_html, f"mobi:{path.name}#{i + 1}", xml=False)
        meta = {
            k: v
            for k, v in {
                **base_meta,
                "title": label,
                "collection": collection,
                "part_index": i + 1,
                "part_count": len(starts),
            }.items()
            if v is not None
        }
        docs.append(ExtractedDoc(path, blocks, meta))

    return docs


# --------------------------------------------------------------------------- #
# DOCX
# --------------------------------------------------------------------------- #


def extract_docx(path: Path) -> ExtractedDoc:
    """Parse a Word document (also what Google Docs exports to)."""
    import docx  # python-docx

    document = docx.Document(str(path))
    blocks: list[Block] = []

    for para in document.paragraphs:
        text = clean_inline(para.text)
        style = ""
        try:
            style = (para.style.name or "").lower()
        except Exception:
            pass

        if not text:
            continue

        if style.startswith("heading") or style in ("title", "subtitle"):
            level = 1
            m = re.search(r"(\d+)", style)
            if m:
                level = 1 if int(m.group(1)) <= 2 else 2
            if style in ("title", "subtitle"):
                level = 1 if style == "title" else 2
            blocks.append(Block(BLOCK_HEADING, clean_heading(text), level=level,
                                origin=path.name))
            continue

        if is_scene_break_line(text):
            blocks.append(Block(BLOCK_SCENE_BREAK, "", origin=path.name))
            continue

        if len(text) <= 70 and _CHAPTER_RE.match(text):
            blocks.append(Block(BLOCK_HEADING, clean_heading(text), level=1,
                                origin=path.name))
            continue

        blocks.append(Block(BLOCK_PARAGRAPH, text, origin=path.name))

    meta_title = None
    try:
        if document.core_properties.title:
            meta_title = clean_heading(document.core_properties.title)
    except Exception:
        pass

    return ExtractedDoc(path, blocks, {"title": meta_title or path.stem})


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #

_MOBI_EXTENSIONS = (".mobi", ".azw", ".azw3", ".prc")
SUPPORTED_EXTENSIONS = (
    ".epub", ".mobi", ".azw", ".azw3", ".prc",
    ".txt", ".md", ".markdown", ".docx", ".html", ".htm", ".xhtml",
)


def extract_file(
    path: Path,
    aggressive_breaks: bool = False,
    numbered_breaks: bool = False,
    strip_pages: bool = False,
    split_omnibus: bool = True,
) -> list[ExtractedDoc]:
    """Dispatch on file extension. Always returns a list of documents.

    Most formats yield exactly one document. A Kindle omnibus yields one per
    novel it contains, so that separate books are never merged into one.
    """
    suffix = path.suffix.lower()
    if suffix == ".epub":
        return extract_epub_collection(path, split_omnibus=split_omnibus)
    if suffix == ".docx":
        return [extract_docx(path)]
    if suffix in (".html", ".htm", ".xhtml"):
        return [extract_html(path)]
    if suffix in (".txt", ".md", ".markdown"):
        return [
            extract_text(
                path,
                aggressive_breaks=aggressive_breaks,
                numbered_breaks=numbered_breaks,
                strip_pages=strip_pages,
            )
        ]
    if suffix in _MOBI_EXTENSIONS:
        return extract_mobi(path, split_omnibus=split_omnibus)
    if suffix == ".pdf":
        raise RuntimeError(
            f"{path.name}: PDF has no reliable paragraph structure. Export the "
            f"source to DOCX/EPUB instead so paragraphs are preserved."
        )
    raise RuntimeError(f"{path.name}: unsupported extension {suffix!r}")


def discover_sources(root: Path, extensions: Iterable[str] = SUPPORTED_EXTENSIONS,
                     recursive: bool = True) -> list[Path]:
    """Find candidate source files under ``root``, skipping junk."""
    exts = {e.lower() for e in extensions}
    it = root.rglob("*") if recursive else root.glob("*")
    found = []
    for p in it:
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        if p.name.startswith(".") or p.name.startswith("~$"):
            continue
        if "__MACOSX" in p.parts:
            continue
        found.append(p)
    return sorted(found)
