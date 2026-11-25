"""Generuje fragmenty LaTeX i wykresy do report/raport.pdf."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from _utils_.log_print import log_print
from _utils_.run_paths import arch_root, run_tag_from_dir_name
from finetune import completed_run_dir
from finetune_presets import FINETUNE_QUEUE, PRETRAINED_LM, UNFREEZE_FRACTIONS, _SCRATCH_DIMS
from finetuning.cls_data import LABEL_NAMES, prepare_cls_data
from finetuning.cls_models import CLS_ARCHITECTURE
from metrics.run_metrics import META_FILENAME, METRICS_FILENAME, TEST_METRICS_FILENAME
from text_and_tokenizer import load_tokenizer
from train import load_checkpoint_payload

REPORT_ROOT = Path(__file__).resolve().parents[1] / "report"
GENERATED = REPORT_ROOT / "generated"
FIGURES = GENERATED / "figures"


def _tex_escape(s: str) -> str:
    s = str(s)
    repl = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_"}
    for a, b in repl.items():
        s = s.replace(a, b)
    s = re.sub(r"[^\x00-\x7F]+", "?", s)
    return s


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _iter_run_dirs() -> list[Path]:
    root = arch_root(CLS_ARCHITECTURE)
    if not root.is_dir():
        return []
    out: list[Path] = []
    for child in root.iterdir():
        if child.is_dir() and not child.name.startswith("_"):
            out.append(child)
    return out


def _latest_runs_by_tag() -> dict[str, Path]:
    by_tag: dict[str, Path] = {}
    for child in _iter_run_dirs():
        tag = run_tag_from_dir_name(child.name)
        if not tag or not (child / TEST_METRICS_FILENAME).is_file():
            continue
        prev = by_tag.get(tag)
        if prev is None or child.name < prev.name:
            by_tag[tag] = child
    return by_tag


def _fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", r"\,")


def _table(caption: str, label: str, colspec: str, header: str, body: list[str]) -> str:
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            rf"\begin{{tabular}}{{{colspec}}}",
            r"\toprule",
            header,
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    ) + "\n"


def generate_pretrain_section() -> None:
    body: list[str] = []
    for lm_tag, meta in PRETRAINED_LM.items():
        pt = Path(__file__).resolve().parents[1] / meta["pt_path"]
        name = lm_tag.replace("dual3060-", "miniLLaMA-")
        params = d_model = heads = d_ff = ppl = epochs = "--"
        layers = str(meta["num_layers"])
        if pt.is_file():
            payload = load_checkpoint_payload(pt, mmap=True)
            hp = payload.get("hparams") or {}
            ppl_v = payload.get("eval_perplexity")
            if ppl_v is not None:
                ppl = f"{float(ppl_v):.1f}"
            d_model = str(hp.get("hidden_size", "--"))
            heads = str(hp.get("num_heads", "--"))
            d_ff = str(hp.get("ff_size", "--"))
            layers = str(hp.get("num_layers", meta["num_layers"]))
            del payload
            meta_path = pt.parent / META_FILENAME
            if meta_path.is_file():
                rm = json.loads(meta_path.read_text(encoding="utf-8"))
                n_params = (rm.get("model") or {}).get("params")
                if n_params:
                    params = _fmt_int(int(n_params))
                epochs = str(rm.get("epochs", "--"))
        body.append(
            f"{name} & {params} & {d_model} & {layers} & {heads} & {d_ff} & {ppl} & {epochs} \\\\"
        )
    _write(
        GENERATED / "pretrain_table.tex",
        _table(
            "Pretrenowane modele miniLLaMA (WikiText-103).",
            "tab:pretrain",
            "@{}lrrrrrrr@{}",
            r"Model & Parametry & $d_\text{model}$ & Bloki & Głowy & $d_\text{ff}$ & Val PPL & Epoki (cel) \\",
            body,
        ),
    )

    proc = (
        r"Korpus: \texttt{Salesforce/wikitext}, konfiguracja \texttt{wikitext-103-raw-v1}. "
        r"Trening na pełnym splicie train (bez limitu \texttt{max\_corpus\_chars}); walidacja na pełnym HF validation. "
        r"Tokenizer: BPE byte-fallback (NFC + Metaspace), słownik 32\,000. "
        r"Architektura \texttt{transformer\_miniLLaMA}: RMSNorm pre-norm, RoPE, SwiGLU, wiązane embeddingi. "
        r"Lightning DDP (\texttt{ddp\_spawn}), 2$\times$RTX 3060, batch 12/GPU (global 24), LR $6\times10^{-4}$, warmup 500 kroków, "
        r"ok.\,9397 kroków/epoka ($\approx$115M tokenów/epoka przy \texttt{seq\_len}=512). "
        r"Checkpointy zamrożone w presetach \texttt{PRETRAINED\_LM} (pełny WikiText, nie smoke)."
    )
    _write(GENERATED / "pretrain_prose.tex", proc + "\n")


def generate_dataset_section() -> None:
    tok = load_tokenizer()
    _, _, _, spec_bpe, summary_bpe = prepare_cls_data(
        max_len=128,
        max_train_examples=0,
        max_val_examples=0,
        max_test_examples=0,
        tokenizer_kind="bpe",
        bpe_tokenizer=tok,
    )
    from transformers import AutoTokenizer

    from finetuning.hf_cache import offline_first

    gpt_tok = offline_first(
        "openai-community/gpt2", AutoTokenizer.from_pretrained, "openai-community/gpt2"
    )
    if gpt_tok.pad_token_id is None:
        gpt_tok.pad_token = gpt_tok.eos_token
    _, _, _, _spec_hf, summary_hf = prepare_cls_data(
        max_len=128,
        max_train_examples=0,
        max_val_examples=0,
        max_test_examples=0,
        tokenizer_kind="hf",
        hf_tokenizer=gpt_tok,
    )
    s = summary_bpe
    dist = s.get("train_class_dist") or {}
    splits = _table(
        "Tweet Eval (sentiment): rozmiary splitów i rozkład klas w zbiorze treningowym.",
        "tab:cls-data",
        "@{}lr@{}",
        r"Cecha & Wartość \\",
        [
            rf"Zbiór HF & \texttt{{{_tex_escape(s['hf_dataset'])}}} / \texttt{{{_tex_escape(s['hf_config'])}}} \\",
            rf"Liczba klas & {s['num_labels']} \\",
            rf"Train / val / test & {_fmt_int(s['train_size'])} / {_fmt_int(s['val_size'])} / {_fmt_int(s['test_size'])} \\",
            *[
                rf"Train: {_tex_escape(k)} & {_fmt_int(v)} ({100.0 * v / s['train_size']:.1f}\%) \\"
                for k, v in dist.items()
            ],
            rf"Maks. długość sekwencji & {s['max_len']} tokenów \\",
        ],
    )
    tokens = _table(
        "Długość tweetów w tokenach (próbka 5\\,000 przykładów treningowych).",
        "tab:cls-tokens",
        "@{}lrrrrr@{}",
        r"Tokenizer & Słownik & Średnio & p95 & Maks. & Ucięte \\",
        [
            rf"BPE (nasze LM) & {_fmt_int(summary_bpe['vocab_size'])} & {summary_bpe['token_len_mean']:.1f} & "
            rf"{summary_bpe['token_len_p95']:.0f} & {summary_bpe['token_len_max']:.0f} & "
            rf"{summary_bpe['truncated_pct_train_sample']:.1f}\% \\",
            rf"GPT-2 & {_fmt_int(summary_hf['vocab_size'])} & {summary_hf['token_len_mean']:.1f} & "
            rf"{summary_hf['token_len_p95']:.0f} & {summary_hf['token_len_max']:.0f} & "
            rf"{summary_hf['truncated_pct_train_sample']:.1f}\% \\",
        ],
    )
    _write(GENERATED / "dataset_table.tex", splits + tokens)
    _write(GENERATED / "dataset_summary.json", json.dumps(summary_bpe, indent=2, ensure_ascii=False))


_MODE_LABEL = {
    "full": "full",
    "lora": "LoRA",
    "head_only": "sama głowa",
    "unfreeze_last": "ostatnie bloki",
}


def _backbone_label(source: str) -> str:
    if source == "scratch":
        return "scratch-5M"
    if source.startswith("hf:"):
        return "GPT-2"
    for lm_tag, meta in PRETRAINED_LM.items():
        if source == f"pt:{meta['pt_path']}":
            return lm_tag.replace("dual3060-", "pt-")
    return "?"


def generate_grid_section() -> None:
    lines = [
        r"{\small\setlength{\tabcolsep}{5pt}\setlength{\LTcapwidth}{\linewidth}",
        r"\begin{longtable}{@{}llllrl@{}}",
        r"\caption{Kolejka fine-tune: backbone, metoda, odmrożone bloki i LR.}\label{tab:grid}\\",
        r"\toprule",
        r"run\_tag & Backbone & Metoda & Bloki & Train & LR \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"run\_tag & Backbone & Metoda & Bloki & Train & LR \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for p in FINETUNE_QUEUE:
        k = p.get("unfreeze_last_n")
        L = p.get("backbone_num_layers")
        if p["finetune_mode"] == "unfreeze_last" and k and L:
            fracs = ", ".join(f"{int(f * 100)}" for f in p.get("unfreeze_fractions") or [])
            blocks = rf"{k}/{L} ({fracs}\%)"
        elif p["finetune_mode"] == "head_only":
            blocks = "0"
        else:
            blocks = "--"
        n_train = int(p["max_train_examples"])
        train = _fmt_int(n_train) if n_train > 0 else "pełny"
        lines.append(
            rf"\texttt{{{_tex_escape(p['run_tag'])}}} & {_backbone_label(str(p['source']))} & "
            rf"{_MODE_LABEL[p['finetune_mode']]} & {blocks} & {train} & "
            rf"{p['baseline_learning_rate']:.0e} \\"
        )
    lines += [r"\end{longtable}", r"}"]
    _write(GENERATED / "grid_table.tex", "\n".join(lines) + "\n")
    prose = (
        f"Eksperyment: {len(FINETUNE_QUEUE)} runów, {int(FINETUNE_QUEUE[0]['epochs'])} epok, "
        f"global batch 64, metody pełny trening / LoRA / sama głowa / ostatnie $k$ bloków "
        f"(ułamki {UNFREEZE_FRACTIONS}, $k=\\lceil pL\\rceil$, bez duplikatów). "
        f"Scratch 5M: od zera ({_SCRATCH_DIMS['num_layers']} bloki). "
        r"Porównanie z \texttt{openai-community/gpt2} (12 warstw)."
    )
    _write(GENERATED / "grid_prose.tex", prose + "\n")


def generate_results_section() -> None:
    rows: list[dict] = []
    missing: list[str] = []
    for preset in FINETUNE_QUEUE:
        tag = preset["run_tag"]
        run_dir = completed_run_dir(preset)
        if run_dir is None:
            missing.append(tag)
            continue
        meta = json.loads((run_dir / META_FILENAME).read_text(encoding="utf-8"))
        test_m = meta.get("test_metrics") or json.loads(
            (run_dir / TEST_METRICS_FILENAME).read_text(encoding="utf-8")
        )
        model_m = meta.get("model") or {}
        rows.append(
            {
                "run_tag": tag,
                "finetune_mode": meta.get("hparams", {}).get("finetune_mode"),
                "params": model_m.get("params"),
                "trainable": model_m.get("trainable"),
                "acc": 100.0 * float(test_m.get("accuracy", 0)),
                "f1m": 100.0 * float(test_m.get("f1_macro", 0)),
                "cls_ppl": test_m.get("cls_perplexity"),
                "lm_after_ppl": test_m.get("lm_after_lm_perplexity_token"),
                "train_s": meta.get("train_total_seconds"),
                "infer_ms": test_m.get("inference_ms_per_example"),
            }
        )

    if rows:
        df = pd.DataFrame(rows).sort_values("acc", ascending=False)
        df.to_csv(GENERATED / "results_summary.csv", index=False)
        lines = [
            r"\begin{longtable}{@{}lrrrrrr@{}}",
            r"\toprule",
            r"run\_tag & acc [\%] & F1 macro [\%] & cls PPL & LM PPL (po) & czas [s] & ms/ex \\",
            r"\midrule",
            r"\endfirsthead",
        ]
        for _, r in df.iterrows():
            lines.append(
                f"{_tex_escape(r['run_tag'])} & {r['acc']:.2f} & {r['f1m']:.2f} & "
                f"{float(r['cls_ppl']):.3f} & {float(r['lm_after_ppl']):.1f} & "
                f"{float(r['train_s']):.0f} & {float(r['infer_ms']):.2f} \\\\"
            )
        lines += [r"\bottomrule", r"\end{longtable}"]
        _write(GENERATED / "results_table.tex", "\n".join(lines) + "\n")

        FIGURES.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.barh(df["run_tag"], df["acc"], color="steelblue")
        ax.set_xlabel("accuracy test [%]")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(FIGURES / "accuracy_ranking.pdf")
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        x = range(len(df))
        ax2.bar(x, df["f1m"], color="darkorange")
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(df["run_tag"], rotation=60, ha="right", fontsize=7)
        ax2.set_ylabel("F1 macro test [%]")
        fig2.tight_layout()
        fig2.savefig(FIGURES / "f1_ranking.pdf")
        plt.close(fig2)
    else:
        _write(
            GENERATED / "results_table.tex",
            r"\noindent\textit{Brak ukończonych runów --- tabela wypełni się po nocnej kolejce.}" + "\n",
        )

    miss_lines = [r"\begin{itemize}"]
    if missing:
        for tag in missing:
            miss_lines.append(f"  \\item {_tex_escape(tag)}")
    else:
        miss_lines.append(r"  \item Wszystkie runy ukończone.")
    miss_lines.append(r"\end{itemize}")
    _write(GENERATED / "missing_runs.tex", "\n".join(miss_lines) + "\n")

    val_curves = []
    for tag, run_dir in _latest_runs_by_tag().items():
        mpath = run_dir / METRICS_FILENAME
        if not mpath.is_file():
            continue
        hist = pd.read_csv(mpath)
        if "eval_f1_macro" not in hist.columns:
            continue
        val_curves.append((tag, hist))
    if val_curves:
        fig, ax = plt.subplots(figsize=(10, 6))
        for tag, hist in val_curves[:20]:
            ax.plot(hist["epoch"], hist["eval_f1_macro"], marker="o", label=tag, alpha=0.8)
        ax.set_xlabel("epoka")
        ax.set_ylabel("val F1 macro")
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(FIGURES / "val_f1_curves.pdf")
        plt.close(fig)


def main() -> Path:
    GENERATED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    generate_pretrain_section()
    generate_dataset_section()
    generate_grid_section()
    generate_results_section()
    log_print(f"Raport: wygenerowano {GENERATED}")
    return GENERATED


if __name__ == "__main__":
    main()