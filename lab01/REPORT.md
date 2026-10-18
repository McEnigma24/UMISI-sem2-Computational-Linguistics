# Lab01: Tokenization Efficiency Benchmark

**Author:** Wojciech Mackowiak  
**Course:** UMISI Computational Linguistics — Lab 01

## 1. Introduction

This report compares three tokenization strategies on the same language modeling task: a pre-trained **GPT-2 BPE** tokenizer, a **whitespace + punctuation** tokenizer with a fixed dictionary and `<UNK>`, and a **SentencePiece Unigram** model trained on WikiText-103 raw. All vocabularies use **N = 50,257** tokens. Models share the same decoder-only GPT architecture and training hyperparameters; only tokenization differs.

## 2. Data

- **Corpus:** [WikiText-103 raw](https://huggingface.co/datasets/Salesforce/wikitext) (`wikitext-103-raw-v1`).
- **Preprocessing:** empty lines removed; `@-@`, `@,@`, `@.@` restored to `-`, `,`, `.`.
- **Splits:** `train` / `valid` for training; **`test`** (~1.3 MB) for all reported metrics and tokenization statistics (held-out from tokenizer training).

**Word** = whitespace-separated token on cleaned text. **Character** = raw UTF-8 character including newlines.

## 3. Tokenizers

| Tokenizer | Type | Notes |
|-----------|------|--------|
| `gpt2` | Pre-trained BPE | Hugging Face `gpt2`; EOS `<\|endoftext\|>` |
| `whitespace` | Word + punctuation | Top-(N−3) types from train; `<PAD>`, `<EOS>`, `<UNK>` |
| `spm` | SentencePiece Unigram | Trained on train split; `vocab_size=50257` |

## 4. Model and training

- **Architecture:** 6-layer pre-LN GPT, `d_model=512`, 8 heads, FFN 2048, context 256, tied embeddings, dropout 0.1 (~45M parameters).
- **Optimizer:** AdamW, lr 6e-4, cosine decay with 2000 warmup steps, weight decay 0.1, grad clip 1.0, fp16 on GPU.
- **Data exposure:** each model trains for the same number of **epochs over the raw training text** (default 1). Optimization steps differ because total token counts differ per tokenizer.

## 5. Evaluation

- **Perplexity:** sliding-window NLL (context 256, stride 128) on the test token stream; reported as **word-level** and **character-level** perplexity (not token-level). Bits-per-character included.
- **OOV:** whitespace tokenizer only — share of words mapped to `<UNK>` on test.
- **Efficiency:** tokenizer training time, encode throughput (tokens/s), LM wall time and peak GPU memory, inference and short-generation latency.

For the whitespace model, `<UNK>` tokens can artificially lower LM perplexity; an optional **UNK-corrected** word perplexity adds a unigram character model penalty for OOV words.

## 6. Discussion (trade-offs)

Fill in after experiments complete. Consider:

- How subword tokenization (GPT-2, SPM) handles rare and compound words vs. whitespace `<UNK>`.
- Whether lower tokens-per-word always improves LM quality at fixed compute.
- Fairness of comparing training steps when token sequences differ in length.
- English WikiText with GPT-2 BPE vs. Polish or other corpora (future work).

## 7. References

- Kudo, T. SentencePiece (EMNLP 2018).
- Sennrich et al. Subword NMT (ACL 2016).
- Mielke, S. [Comparing perplexities](https://sjmielke.com/comparing-perplexities.htm).

<!-- AUTO-GENERATED -->

# Lab01 Tokenization Benchmark — generated summary

## Model and training defaults

- Architecture: GPT decoder-only (pre-LN), defaults {'n_layer': 6, 'n_head': 8, 'n_embd': 512, 'n_ff': 2048, 'block_size': 256, 'dropout': 0.1, 'vocab_size': 50257}
- Optimizer: AdamW, {'batch_size': 64, 'lr': 0.0006, 'weight_decay': 0.1, 'betas': (0.9, 0.95), 'grad_clip': 1.0, 'warmup_steps': 2000, 'max_epochs': 1, 'eval_interval': 500, 'log_interval': 50, 'seed': 42, 'use_amp': True}

## Perplexity (test split, word/char level)

| Tokenizer | Word PPL | Char PPL | BPC | Word PPL (UNK-corrected) |
|-----------|----------|----------|-----|---------------------------|
| gpt2 | 77.41 | 2.28 | 1.1881 | 77.41 |
| whitespace | 49.68 | 2.10 | 1.0670 | 146.23 |
| spm | 77.91 | 2.28 | 1.1899 | 77.91 |

## Tokenization statistics (test)

| Tokenizer | Tokens/word | Direct words % | OOV % (whitespace) | Encode tok/s |
|-----------|-------------|----------------|--------------------|--------------|
| gpt2 | 1.341 | 74.5 | n/a | 350290 |
| whitespace | 1.010 | 95.7 | 4.31 | 2828853 |
| spm | 1.080 | 94.8 | n/a | 2596726 |

## Training efficiency

| Tokenizer | Wall time (s) | Train tok/s | Peak GPU MB |
|-----------|---------------|-------------|-------------|
| gpt2 | 1201.3 | 95303 | 13917 |
| whitespace | 1055.7 | 98377 | 13917 |
| spm | 1154.1 | 95367 | 13917 |

## Tokenizer training time (seconds)

- gpt2: 1.29s
- whitespace: 34.57s
- spm: 113.14s

## Hardware snapshot

- Host: ag0001
- Torch: 2.14.1+cu126
- CUDA device: Tesla V100-SXM2-32GB

## Qualitative

See [results/qualitative.md](results/qualitative.md).
