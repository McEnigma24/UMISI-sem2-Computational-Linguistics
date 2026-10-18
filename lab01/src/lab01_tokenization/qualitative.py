"""Write qualitative tokenization comparison."""

from __future__ import annotations

from lab01_tokenization.config import RESULTS, TOKENIZER_NAMES
from lab01_tokenization.samples import qualitative_samples
from lab01_tokenization.tokenizers_impl import load_tokenizer
from lab01_tokenization.utils import count_words


def run_qualitative() -> None:
    samples = qualitative_samples()
    lines: list[str] = ["# Qualitative tokenization samples", ""]
    for sample in samples:
        text = sample["text"]
        n_words = count_words(text)
        lines.append(f"## {sample['id']}")
        lines.append(f"*{sample['description']}* ({n_words} words)")
        lines.append("")
        lines.append(f"> {text}")
        lines.append("")
        for name in TOKENIZER_NAMES:
            tok = load_tokenizer(name)
            pieces = tok.tokenize(text)
            n_tok = len(pieces)
            words = text.split()
            direct_words = sum(1 for w in words if tok.is_direct(w))
            tpw = n_tok / n_words if n_words else 0.0
            direct_pct = 100.0 * direct_words / n_words if n_words else 0.0
            joined = " | ".join(pieces[:200])
            if len(pieces) > 200:
                joined += " | ..."
            lines.append(f"### {name}")
            lines.append(f"- tokens: {n_tok}, tokens/word: {tpw:.3f}")
            lines.append(f"- direct words: {direct_pct:.1f}%")
            lines.append(f"- tokenized: `{joined}`")
            lines.append("")
        lines.append("---")
        lines.append("")

    out = RESULTS / "qualitative.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
