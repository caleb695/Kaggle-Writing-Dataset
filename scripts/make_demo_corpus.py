#!/usr/bin/env python3
"""Generate a realistic multi-book demo corpus to exercise the whole pipeline.

Produces the same folder shape as the real manuscript library -- one folder per
series, one file per book -- plus the two folders that must be excluded, so you
can confirm the exclusions work before pointing the builder at real sources.

    python scripts/make_demo_corpus.py --root data/demo/raw
    python -m writing_dataset.cli --raw-root data/demo/raw \
        --out-root data/demo/out --run-name demo
"""

from __future__ import annotations

import argparse
import random
import zipfile
from pathlib import Path

SUBJECTS = (
    "dragon wing scale fire shadow mountain river storm silver ember ash tower gate "
    "stone banner hollow crown raven frost lantern tide harbor meadow thorn cinder "
    "willow arrow compass cave sunglass keeper watcher wanderer blade sigil rune "
    "kestrel heron marl tidepool saltglass brightwater farfield"
).split()

SERIES = {
    "Fablehaven and dragonwatch": ["Fablehaven", "Dragonwatch"],
    "five kingdoms and beyonders": ["Sky Raiders", "Rogue Knight", "Crystal Keepers"],
    "Harry Potter": ["Philosophers Stone", "Chamber of Secrets"],
    "micheal vey": ["The Prisoner of Cell 25", "Rise of the Elgen"],
    "unwanteds": ["The Unwanteds", "Island of Silence"],
    "the false prince": ["The False Prince"],
    "greek and roman mythology": ["Myths of Olympus"],
}
EXCLUDED_SERIES = {
    "Wings of Fire Arc 1": ["The Dragonet Prophecy", "The Lost Heir"],
    "Wings of Fire Arc 2": ["Moon Rising", "Winter Turning"],
    "Onyx": ["Onyx"],
}


def paragraph(rnd: random.Random, n_words: int) -> str:
    words = [rnd.choice(SUBJECTS) for _ in range(n_words)]
    words[0] = words[0].capitalize()
    text = " ".join(words)
    return text + rnd.choice([".", ".", ".", "!", "?"])


def make_chapters(rnd: random.Random, n_chapters: int):
    chapters = []
    for c in range(n_chapters):
        scenes = []
        for _ in range(rnd.randint(2, 5)):
            paragraphs = [paragraph(rnd, rnd.randint(40, 180)) for _ in range(rnd.randint(4, 22))]
            scenes.append(paragraphs)
        chapters.append((f"Chapter {c + 1}", scenes))
    return chapters


def write_epub(path: Path, title: str, chapters) -> None:
    """Write a valid EPUB (mimetype first, stored) with the given chapters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    names = []
    for i, (chapter_title, scenes) in enumerate(chapters):
        body = [f"<h2>{chapter_title}</h2>"]
        for si, paragraphs in enumerate(scenes):
            if si:
                body.append("<hr/>")
            body.extend(f"<p>{p}</p>" for p in paragraphs)
        name = f"chap_{i:03d}.xhtml"
        names.append((name, chapter_title))
        (path.parent / name).write_text(
            '<?xml version="1.0" encoding="utf-8"?>'
            '<html xmlns="http://www.w3.org/1999/xhtml"><head>'
            f"<title>{chapter_title}</title></head><body>{''.join(body)}</body></html>",
            encoding="utf-8",
        )

    manifest = "\n".join(
        f'<item id="c{i}" href="{n}" media-type="application/xhtml+xml"/>'
        for i, (n, _) in enumerate(names)
    )
    spine = "\n".join(f'<itemref idref="c{i}"/>' for i in range(len(names)))
    opf = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
        f"<dc:title>{title}</dc:title><dc:language>en</dc:language>"
        '<dc:identifier id="uid">urn:uuid:demo</dc:identifier></metadata>'
        f'<manifest><item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" '
        f'properties="nav"/>\n{manifest}</manifest><spine>{spine}</spine></package>'
    )
    container = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        '<rootfiles><rootfile full-path="content.opf" '
        'media-type="application/oebps-package+xml"/></rootfiles></container>'
    )
    nav = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml" '
        'xmlns:epub="http://www.idpf.org/2007/ops"><head><title>Contents</title></head>'
        '<body><nav epub:type="toc" id="toc"><ol>'
        + "".join(f'<li><a href="{n}">{t}</a></li>' for n, t in names)
        + "</ol></nav></body></html>"
    )

    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip", zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", container, zipfile.ZIP_DEFLATED)
        zf.writestr("content.opf", opf, zipfile.ZIP_DEFLATED)
        zf.writestr("nav.xhtml", nav, zipfile.ZIP_DEFLATED)
        for name, _ in names:
            zf.write(path.parent / name, name)
            (path.parent / name).unlink()


def write_txt(path: Path, title: str, chapters) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [title, ""]
    for chapter_title, scenes in chapters:
        lines += [chapter_title, ""]
        for si, paragraphs in enumerate(scenes):
            if si:
                lines += ["* * *", ""]
            for p in paragraphs:
                lines += [p, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("data/demo/raw"))
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--min-chapters", type=int, default=10)
    ap.add_argument("--max-chapters", type=int, default=22)
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    written = 0

    for series, books in SERIES.items():
        for i, title in enumerate(books):
            n = rnd.randint(args.min_chapters, args.max_chapters)
            chapters = make_chapters(rnd, n)
            suffix = ".txt" if (i % 4 == 3) else ".epub"
            path = args.root / series / f"{title}{suffix}"
            if suffix == ".epub":
                write_epub(path, title, chapters)
            else:
                write_txt(path, title, chapters)
            written += 1

    for series, books in EXCLUDED_SERIES.items():
        for title in books:
            chapters = make_chapters(rnd, rnd.randint(8, 12))
            write_epub(args.root / series / f"{title}.epub", title, chapters)
            written += 1

    total_bytes = sum(f.stat().st_size for f in args.root.rglob("*") if f.is_file())
    print(f"wrote {written} books under {args.root} ({total_bytes / 1e6:.1f} MB)")
    print(f"\nnext:\n  python -m writing_dataset.cli --raw-root {args.root} "
          f"--out-root data/demo/out --run-name demo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
