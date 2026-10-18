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
