"""Fixed qualitative sample texts (each >= 30 words)."""

from __future__ import annotations

from lab01_tokenization.data import iter_split_lines
from lab01_tokenization.utils import count_words

SAMPLE_MODERN = (
    "Engineers deployed ChatGPT-style assistants across hospitals during COVID-19, "
    "routing patient questions through encrypted APIs at https://example.org/health. "
    "Stakeholders demanded transparent benchmarking, lower latency, and reproducible "
    "tokenization pipelines before approving production rollouts in late 2024."
)

SAMPLE_RARE = (
    "The paleontologist unearthed a Psittacosaurus specimen near Ulaanbaatar, measuring "
    "2.4 metres and dated to approximately 120 million years before present. "
    "Spectroscopy revealed iridium anomalies consistent with Cretaceous-Paleogene boundary "
    "hypotheses debated in peer-reviewed journals."
)


def wikitext_excerpt(min_words: int = 30) -> str:
    fallback: list[str] = []
    for line in iter_split_lines("test"):
        if count_words(line) >= min_words:
            return line
        fallback.append(line)
        if len(fallback) >= 5:
            break
    return " ".join(fallback)


def qualitative_samples() -> list[dict[str, str]]:
    samples = [
        {"id": "wikitext_test", "description": "Held-out WikiText-103 test line", "text": wikitext_excerpt()},
        {"id": "modern_ood", "description": "Modern out-of-domain text", "text": SAMPLE_MODERN},
        {"id": "rare_terms", "description": "Rare names, numbers, and scientific terms", "text": SAMPLE_RARE},
    ]
    for s in samples:
        n = count_words(s["text"])
        assert n >= 30, f"sample {s['id']} has only {n} words"
    return samples
