"""Evaluate perplexity with sliding-window NLL."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from lab01_tokenization.config import DATA_RAW, EvalConfig, ModelConfig, RESULTS, paths_for_tokenizer
from lab01_tokenization.data import load_split_text
from lab01_tokenization.model import GPT
from lab01_tokenization.tokenizers_impl import load_tokenizer
from lab01_tokenization.tokenizers_impl.whitespace import WhitespaceTokenizer
from lab01_tokenization.utils import (
    build_char_unigram,
    count_chars,
    count_words,
    iter_words,
    read_json,
    timer,
    unk_char_nll,
    write_json,
)


def _load_model(tokenizer_name: str, device: torch.device) -> tuple[GPT, ModelConfig]:
    paths = paths_for_tokenizer(tokenizer_name)
    ckpt_path = paths["ckpt_dir"] / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = paths["ckpt_dir"] / "last.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"no checkpoint for {tokenizer_name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = ModelConfig(**ckpt["model_cfg"])
    model = GPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, model_cfg


def sliding_window_nll(
    model: GPT,
    tokens: np.ndarray,
    cfg: EvalConfig,
    device: torch.device,
) -> float:
    nll = 0.0
    stride = cfg.stride
    ctx = cfg.block_size
    ids = tokens.astype(np.int64)
    L = len(ids)
    if L < 2:
        return 0.0

    starts = list(range(0, L - 1, stride))
    with torch.no_grad():
        for si, begin in enumerate(starts):
            end = min(begin + ctx + 1, L)
            chunk = ids[begin:end]
            if len(chunk) < 2:
                continue
            inp = torch.tensor(chunk[:-1], device=device).unsqueeze(0)
            tgt = torch.tensor(chunk[1:], device=device)
            with autocast(enabled=device.type == "cuda"):
                logits, _ = model(inp)
            log_probs = F.log_softmax(logits[0], dim=-1)
            token_nll = -log_probs[torch.arange(len(tgt), device=device), tgt]

            if si == 0:
                use = token_nll
            else:
                use = token_nll[-stride:]
            nll += float(use.sum().item())
    return nll


def _unk_correction_nll(tokenizer_name: str, test_text: str) -> float:
    if tokenizer_name != "whitespace":
        return 0.0
    tok = load_tokenizer(tokenizer_name)
    if not isinstance(tok, WhitespaceTokenizer):
        return 0.0
    train_text = (DATA_RAW / "train.txt").read_text(encoding="utf-8")
    char_probs = build_char_unigram(train_text)
    extra = 0.0
    for word in iter_words(test_text):
        if not tok.is_direct(word):
            extra += unk_char_nll(word, char_probs)
    return extra


def _generation_latency(model: GPT, model_cfg: ModelConfig, tokenizer_name: str, device: torch.device) -> dict:
    tok = load_tokenizer(tokenizer_name)
    prompt = "The history of the city began when "
    ids_list = tok.encode(prompt)
    if not ids_list:
        ids_list = [tok.eos_id]
    ids = torch.tensor([ids_list], device=device, dtype=torch.long)
    if ids.size(1) > model_cfg.block_size:
        ids = ids[:, -model_cfg.block_size :]
    t0 = timer()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=32)
    elapsed = timer() - t0
    new_tokens = out.size(1) - ids.size(1)
    return {
        "prompt": prompt,
        "max_new_tokens": 32,
        "new_tokens": int(new_tokens),
        "seconds": elapsed,
        "tokens_per_second": new_tokens / elapsed if elapsed else 0.0,
    }


def evaluate_tokenizer(tokenizer_name: str) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = paths_for_tokenizer(tokenizer_name)
    test_text = load_split_text("test")
    test_tokens = np.memmap(paths["bin_test"], dtype=np.uint16, mode="r")

    model, model_cfg = _load_model(tokenizer_name, device)
    eval_cfg = EvalConfig(block_size=model_cfg.block_size)

    t0 = timer()
    nll = sliding_window_nll(model, np.array(test_tokens), eval_cfg, device)
    infer_sec = timer() - t0

    W = count_words(test_text)
    C = count_chars(test_text)
    word_ppl = math.exp(nll / W) if W else float("inf")
    char_ppl = math.exp(nll / C) if C else float("inf")
    bpc = (nll / math.log(2)) / C if C else float("inf")

    unk_extra = _unk_correction_nll(tokenizer_name, test_text)
    word_ppl_corrected = math.exp((nll + unk_extra) / W) if W else word_ppl

    gen_latency = _generation_latency(model, model_cfg, tokenizer_name, device)

    metrics = {
        "tokenizer": tokenizer_name,
        "nll_nats": nll,
        "word_perplexity": word_ppl,
        "char_perplexity": char_ppl,
        "bits_per_char": bpc,
        "word_perplexity_unk_corrected": word_ppl_corrected,
        "unk_extra_nll": unk_extra,
        "num_words": W,
        "num_chars": C,
        "inference_seconds": infer_sec,
        "generation": gen_latency,
        "eval_block_size": eval_cfg.block_size,
        "eval_stride": eval_cfg.stride,
    }
    write_json(RESULTS / f"eval_{tokenizer_name}.json", metrics)
    train_metrics = read_json(paths["metrics"]) if paths["metrics"].exists() else {}
    write_json(paths["metrics"], {**train_metrics, **metrics})
    print(
        f"eval {tokenizer_name}: word_ppl={word_ppl:.2f} char_ppl={char_ppl:.2f} bpc={bpc:.4f}"
    )
    return metrics
