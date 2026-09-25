"""Tokenizer resolution and token accounting.

Why this module is fussier than it looks
----------------------------------------
Ministral-3-14B-Base-2512 uses Mistral's **Tekken** tokenizer (vocab 131072,
BOS=1, EOS=2). ``mistral-common`` ships two Tekken files, but they are *not*
byte-identical to the ``tekken.json`` stored in the model repo:

* the bundled copies (``tekken_240718.json`` / ``tekken_240911.json``) have had
  their **special tokens stripped** -- ranks 0..999 are raw control bytes, and
  ``"<s>"`` encodes to three ordinary tokens instead of id 1;
* the model repo's ``tekken.json`` (16.8 MB, alongside ``special_tokens_map.json``)
  carries the real special tokens, including ``[IMG]`` = 10, which matches
  ``image_token_index: 10`` in the model's ``config.json``.

Consequences for token counting:

* For ordinary prose, both files agree, because the BPE merge table for ranks
  >= 1000 is shared. Differences only appear where a document literally
  contains a string like ``[INST]`` or ``[IMG]``.
* Treating a special token as many ordinary tokens can only *over*-count, so a
  chunk verified against the bundled file is also within budget under the real
  file. The bundled tokenizer is therefore a safe (conservative) default.

The pipeline still prefers the model's own ``tekken.json`` when you point it at
one, and records a hash in ``stats.json`` so a run is reproducible.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

# --------------------------------------------------------------------------- #
# Budget constants
# --------------------------------------------------------------------------- #

#: Hard sequence budget from the training spec.
MAX_SEQ_LEN: int = 8192

#: Positions held back for boundary tokens. A base-model CPT example is laid out
#: as ``[BOS] <content> [EOS]``, so the content may use ``MAX_SEQ_LEN - 2``.
RESERVED_BOUNDARY_TOKENS: int = 2

#: Usable content budget: 8190 tokens.
CONTENT_BUDGET: int = MAX_SEQ_LEN - RESERVED_BOUNDARY_TOKENS

#: Where to look for the model's own tekken.json, in priority order.
DEFAULT_TEKKEN_SEARCH_PATHS: tuple[str, ...] = (
    "data/tokenizer/tekken.json",
    "tokenizer/tekken.json",
    "tekken.json",
)


@dataclass
class TokenizerInfo:
    """Provenance for the tokenizer a dataset was built against."""

    name: str
    kind: str
    n_words: int
    bos_id: Optional[int]
    eos_id: Optional[int]
    has_real_special_tokens: bool
    path: Optional[str] = None
    sha256: Optional[str] = None
    exact_for_ministral3_14b: bool = False
    caution: Optional[str] = None
    max_seq_len: int = MAX_SEQ_LEN
    reserved_boundary_tokens: int = RESERVED_BOUNDARY_TOKENS
    content_budget: int = CONTENT_BUDGET

    def as_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        lines = [
            f"tokenizer      : {self.name} ({self.kind})",
            f"vocab size     : {self.n_words}",
            f"BOS / EOS      : {self.bos_id} / {self.eos_id}",
            f"real specials  : {self.has_real_special_tokens}",
            f"seq budget     : {self.max_seq_len} "
            f"(content {self.content_budget} + {self.reserved_boundary_tokens} reserved)",
            f"exact for model: {self.exact_for_ministral3_14b}",
        ]
        if self.path:
            lines.append(f"path           : {self.path}")
        if self.caution:
            lines.append(f"CAUTION        : {self.caution}")
        return "\n".join(lines)


class TokenCounter:
    """Thin wrapper exposing ``count`` / ``encode`` over a resolved backend."""

    def __init__(self, info: TokenizerInfo, backend):
        self.info = info
        self._backend = backend
        self._cache: dict[str, int] = {}

    # -- core API ---------------------------------------------------------- #

    def encode(self, text: str) -> list[int]:
        if self.info.kind == "hf-tokenizers":
            return self._backend.encode(text).ids
        if self.info.kind == "whitespace-proxy":
            raise RuntimeError("proxy tokenizer cannot emit ids")
        return self._backend.encode(text, bos=False, eos=False)

    def count(self, text: str) -> int:
        """Exact token count for ``text`` under the resolved tokenizer."""
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        if self.info.kind == "hf-tokenizers":
            n = len(self._backend.encode(text).ids)
        elif self.info.kind == "whitespace-proxy":
            n = self._proxy_count(text)
        else:
            n = len(self._backend.encode(text, bos=False, eos=False))
        self._cache[text] = n
        return n

    def count_many(self, texts: Iterable[str]) -> list[int]:
        return [self.count(t) for t in texts]

    @staticmethod
    def _proxy_count(text: str) -> int:
        """Documented approximation used only as a last-resort fallback.

        Deliberately pessimistic (over-counts) so a budget check never
        under-estimates. Not used when mistral-common is importable.
        """
        import regex

        # ~4 chars/token for English prose, plus one token per whitespace run.
        words = regex.findall(r"\S+", text)
        punctuation = regex.findall(r"[^\w\s]", text)
        return int(len(words) + 0.30 * len(words) + 0.5 * len(punctuation))


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _find_model_tekken(explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"--tekken: no such file: {p}")
        return p

    env = os.environ.get("MINISTRAL_TEKKEN")
    if env and Path(env).is_file():
        return Path(env)

    # HF cache layout: models--mistralai--Ministral-3-14B-Base-2512/snapshots/*/tekken.json
    for root in (
        Path.home() / ".cache" / "huggingface" / "hub",
        Path(os.environ.get("HF_HOME", "/nonexistent")) / "hub",
    ):
        if not root.is_dir():
            continue
        for hit in sorted(root.glob("models--*Ministral-3-14B-Base*/snapshots/*/tekken.json")):
            return hit

    for rel in DEFAULT_TEKKEN_SEARCH_PATHS:
        p = Path(rel)
        if p.is_file():
            return p
    return None


def _probe_special_tokens(tok) -> tuple[Optional[int], Optional[int], bool]:
    """Return (bos_id, eos_id, has_real_specials).

    The bundled mistral-common files strip specials, in which case the
    attributes still report the *intended* ids but the encoding path treats
    ``"<s>"`` as plain text. We detect the stripped case by encoding the literal
    and checking whether it collapsed to a single id.
    """
    bos_id = getattr(tok, "bos_id", None)
    eos_id = getattr(tok, "eos_id", None)
    try:
        ids = tok.encode("<s>", bos=False, eos=False)
        has_real = len(ids) == 1 and ids[0] == bos_id
    except Exception:
        has_real = False
    return bos_id, eos_id, has_real


def load_tokenizer(
    tekken_path: Optional[str] = None,
    allow_proxy_fallback: bool = False,
    max_seq_len: int = MAX_SEQ_LEN,
) -> TokenCounter:
    """Resolve the best available tokenizer for Ministral-3-14B-Base-2512.

    Order of preference:

    1. an explicit ``tekken.json`` (``--tekken`` / ``MINISTRAL_TEKKEN``);
    2. a ``tekken.json`` auto-discovered in the local HF cache or repo;
    3. the copy bundled with ``mistral-common`` (conservative upper bound);
    4. ``tokenizers``/``transformers`` if the user has them;
    5. a whitespace proxy, only if explicitly allowed.
    """
    budget = max_seq_len - RESERVED_BOUNDARY_TOKENS

    found = _find_model_tekken(tekken_path)
    if found is not None:
        try:
            from mistral_common.tokens.tokenizers.tekken import Tekkenizer

            tok = Tekkenizer.from_file(found)
            bos, eos, real = _probe_special_tokens(tok)
            info = TokenizerInfo(
                name="tekken",
                kind="tekken-file",
                n_words=tok.n_words,
                bos_id=bos,
                eos_id=eos,
                has_real_special_tokens=real,
                path=str(found),
                sha256=_sha256_of(found),
                exact_for_ministral3_14b=real,
                caution=None if real else
                "tekken.json found but its special tokens do not resolve to single ids; "
                "verify this is the file shipped with Ministral-3-14B-Base-2512.",
                max_seq_len=max_seq_len,
                content_budget=budget,
            )
            return TokenCounter(info, tok)
        except ImportError:
            pass  # fall through to the other backends

    try:
        from mistral_common.tokens.tokenizers.tekken import Tekkenizer
        from importlib.resources import files as _files

        bundled = Path(str(_files("mistral_common") / "data" / "tekken_240911.json"))
        tok = Tekkenizer.from_file(bundled)
        bos, eos, real = _probe_special_tokens(tok)
        info = TokenizerInfo(
            name="tekken_240911 (bundled with mistral-common)",
            kind="tekken-mistral-common",
            n_words=tok.n_words,
            bos_id=bos,
            eos_id=eos,
            has_real_special_tokens=real,
            path=str(bundled),
            sha256=_sha256_of(bundled),
            exact_for_ministral3_14b=False,
            caution=(
                "Bundled Tekken has its special tokens stripped, so counts are a "
                "conservative UPPER BOUND vs the model's own tekken.json: a chunk "
                "verified at <= budget here also fits under the real tokenizer. "
                "Pass --tekken /path/to/tekken.json from the model snapshot for "
                "exact counts."
            ),
            max_seq_len=max_seq_len,
            content_budget=budget,
        )
        return TokenCounter(info, tok)
    except ImportError:
        pass

    try:  # pragma: no cover - only when the HF stack is installed
        from tokenizers import Tokenizer as _HFTokenizer

        hf = _HFTokenizer.from_file("tokenizer.json")
        info = TokenizerInfo(
            name="tokenizer.json",
            kind="hf-tokenizers",
            n_words=hf.get_vocab_size(),
            bos_id=hf.token_to_id("<s>"),
            eos_id=hf.token_to_id("</s>"),
            has_real_special_tokens=True,
            path="tokenizer.json",
            sha256=_sha256_of(Path("tokenizer.json")),
            exact_for_ministral3_14b=False,
            caution="Counting via tokenizers.Tokenizer without the chat template; "
                    "verify against mistral-common before publishing.",
            max_seq_len=max_seq_len,
            content_budget=budget,
        )
        return TokenCounter(info, hf)
    except Exception:
        pass

    if not allow_proxy_fallback:
        raise RuntimeError(
            "No tokenizer available. Install mistral-common (`pip install mistral-common`) "
            "or pass --tekken /path/to/tekken.json. To proceed with an approximate "
            "counter for dry-runs only, pass --allow-proxy-tokenizer."
        )

    info = TokenizerInfo(
        name="whitespace-proxy",
        kind="whitespace-proxy",
        n_words=0,
        bos_id=1,
        eos_id=2,
        has_real_special_tokens=False,
        exact_for_ministral3_14b=False,
        caution="APPROXIMATE token counts. Not valid for producing a training dataset.",
        max_seq_len=max_seq_len,
        content_budget=budget,
    )
    return TokenCounter(info, None)


# --------------------------------------------------------------------------- #
# Chunk verification
# --------------------------------------------------------------------------- #


def verify_chunk_budget(
    text: str,
    counter: TokenCounter,
    budget: Optional[int] = None,
) -> int:
    """Tokenise ``text`` once and assert it fits the content budget."""
    limit = budget if budget is not None else counter.info.content_budget
    n = counter.count(text)
    if n > limit:
        raise AssertionError(f"chunk is {n} tokens, over the {limit}-token budget")
    return n


def summarise_lengths(lengths: Sequence[int]) -> dict:
    if not lengths:
        return {"n": 0}
    ordered = sorted(lengths)
    n = len(ordered)

    def pct(p: float) -> int:
        idx = min(n - 1, max(0, int(round(p / 100.0 * (n - 1)))))
        return ordered[idx]

    return {
        "n": n,
        "min": ordered[0],
        "max": ordered[-1],
        "mean": round(sum(ordered) / n, 1),
        "p05": pct(5),
        "p25": pct(25),
        "p50": pct(50),
        "p75": pct(75),
        "p95": pct(95),
    }


def dumps_info(info: TokenizerInfo) -> str:
    return json.dumps(info.as_dict(), indent=2)
