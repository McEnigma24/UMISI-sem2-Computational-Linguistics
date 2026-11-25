"""Generuje fragmenty LaTeX i wykresy do report/raport.pdf."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from _utils_.log_print import log_print
from finetune import completed_run_dir
from finetune_presets import FINETUNE_QUEUE, PRETRAINED_LM, UNFREEZE_FRACTIONS, _SCRATCH_DIMS
from finetuning.cls_data import LABEL_NAMES, prepare_cls_data
from metrics.run_metrics import META_FILENAME, METRICS_FILENAME, TEST_METRICS_FILENAME
from text_and_tokenizer import load_tokenizer
from train import load_checkpoint_payload

REPORT_ROOT = Path(__file__).resolve().parents[1] / "report"
GENERATED = REPORT_ROOT / "generated"
FIGURES = GENERATED / "figures"


def _tex_escape(s: str, *, keep_unicode: bool = False) -> str:
    s = str(s)
    repl = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_"}
    for a, b in repl.items():
        s = s.replace(a, b)
    if not keep_unicode:
        s = re.sub(r"[^\x00-\x7F]+", "?", s)
    return s


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
        params = d_model = heads = d_ff = ppl = epochs = minutes = "--"
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
                if rm.get("train_total_seconds"):
                    minutes = f"{float(rm['train_total_seconds']) / 60:.0f}"
        body.append(
            f"{name} & {params} & {d_model} & {layers} & {heads} & {d_ff} & {ppl} & {epochs} & {minutes} \\\\"
        )
    _write(
        GENERATED / "pretrain_table.tex",
        _table(
            "Pretrenowane modele miniLLaMA (WikiText-103). Modele 50M i 110M zatrzymał limit "
            "czasu 90\\,min (ok.\\ 2{,}25 i 1{,}3 epoki zamiast 3).",
            "tab:pretrain",
            "@{}lrrrrrrrr@{}",
            r"Model & Parametry & $d_\text{model}$ & Bloki & Głowy & $d_\text{ff}$ & Val PPL & Epoki (cel) & Czas [min] \\",
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
    train_ds, _, test_ds, spec_bpe, summary_bpe = prepare_cls_data(
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
    split_rows = {"train": train_ds.rows, "test": test_ds.rows}
    dist_body = []
    for name in LABEL_NAMES:
        cells = []
        for rows in split_rows.values():
            n = sum(1 for r in rows if LABEL_NAMES[int(r["label"])] == name)
            cells.append(f"{_fmt_int(n)} ({100.0 * n / len(rows):.1f}\\%)")
        val_n = int((s.get("val_class_dist") or {}).get(name, 0))
        cells.insert(1, f"{_fmt_int(val_n)} ({100.0 * val_n / s['val_size']:.1f}\\%)")
        dist_body.append(rf"\texttt{{{name}}} & " + " & ".join(cells) + r" \\")
    dist_body.append(r"\midrule")
    dist_body.append(
        rf"razem & {_fmt_int(s['train_size'])} & {_fmt_int(s['val_size'])} & {_fmt_int(s['test_size'])} \\"
    )
    splits = _table(
        rf"Tweet Eval (\texttt{{{_tex_escape(s['hf_dataset'])}}}, konfiguracja \texttt{{{s['hf_config']}}}): "
        r"oficjalne splity i rozkład klas. Rozkład testu wyraźnie różni się od treningu.",
        "tab:cls-data",
        "@{}lrrr@{}",
        r"Klasa & Train & Val & Test \\",
        dist_body,
    )
    examples = []
    for name in LABEL_NAMES:
        picked = [r for r in train_ds.rows if LABEL_NAMES[int(r["label"])] == name][:2]
        for r in picked:
            text = _tex_escape(r["text"].replace("\n", " ")[:120]).replace("@", r"@\allowbreak{}")
            examples.append(rf"\texttt{{{name}}} & {text} \\")
    examples_tex = _table(
        "Przykładowe tweety z zbioru treningowego (pierwsze dwa z każdej klasy; znaki spoza ASCII zastąpione ?).",
        "tab:cls-examples",
        r"@{}lp{0.78\linewidth}@{}",
        r"Klasa & Tekst \\",
        examples,
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
    _write(GENERATED / "dataset_examples.tex", examples_tex)
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


_BACKBONE_ORDER = ["scratch-5M", "pt-5M", "pt-10M", "pt-25M", "pt-50M", "pt-110M", "GPT-2"]
_BACKBONE_COLOR = {
    "scratch-5M": "#9e9e9e",
    "pt-5M": "#c6dbef",
    "pt-10M": "#9ecae1",
    "pt-25M": "#6baed6",
    "pt-50M": "#3182bd",
    "pt-110M": "#08519c",
    "GPT-2": "#e6550d",
}
_METHOD_COLS = ["full", "LoRA", "sama głowa", "10%", "20%", "30%", "50%"]


def _peft_lm_head_copy_numel(source: str) -> int:
    """PEFT ``modules_to_save=["head"]`` dopasowuje też ``backbone.lm_head`` (sufiks) i dokłada jego
    trenowalną kopię (vocab×d). Nie leży ona na ścieżce klasyfikacji, więc nie dostaje gradientu —
    odejmujemy ją od liczby parametrów, żeby pokazać faktycznie trenowane wagi LoRA."""
    pt = Path(__file__).resolve().parents[1] / source.removeprefix("pt:")
    payload = load_checkpoint_payload(pt, mmap=True)
    hp = payload.get("hparams") or {}
    n = int(hp["vocab_size"]) * int(hp["hidden_size"])
    del payload
    return n


def _collect_results() -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict] = []
    missing: list[str] = []
    for preset in FINETUNE_QUEUE:
        tag = preset["run_tag"]
        run_dir = completed_run_dir(preset)
        if run_dir is None:
            missing.append(tag)
            continue
        meta = json.loads((run_dir / META_FILENAME).read_text(encoding="utf-8"))
        test_m = json.loads((run_dir / TEST_METRICS_FILENAME).read_text(encoding="utf-8"))
        model_m = meta.get("model") or {}
        mode = preset["finetune_mode"]
        params = int(model_m.get("params") or 0)
        trainable = int(model_m.get("trainable") or 0)
        source = str(preset["source"])
        if mode == "lora" and source.startswith("pt:"):
            dup = _peft_lm_head_copy_numel(source)
            params -= dup
            trainable -= dup
        rows.append(
            {
                "run_tag": tag,
                "run_dir": str(run_dir),
                "backbone": _backbone_label(str(preset["source"])),
                "mode": mode,
                "method": _MODE_LABEL[mode],
                "k": preset.get("unfreeze_last_n"),
                "L": preset.get("backbone_num_layers"),
                "fractions": list(preset.get("unfreeze_fractions") or []),
                "n_train": int(preset["max_train_examples"]),
                "epochs": int(preset["epochs"]),
                "params": params,
                "trainable": trainable,
                "acc": 100.0 * float(test_m["accuracy"]),
                "f1m": 100.0 * float(test_m["f1_macro"]),
                "f1w": 100.0 * float(test_m["f1_weighted"]),
                "cls_ppl": float(test_m["cls_perplexity"]),
                "lm_before": float(test_m.get("lm_before_lm_perplexity_token", float("nan"))),
                "lm_after": float(test_m.get("lm_after_lm_perplexity_token", float("nan"))),
                "bpc_before": float(test_m.get("lm_before_lm_bits_per_char", float("nan"))),
                "bpc_after": float(test_m.get("lm_after_lm_bits_per_char", float("nan"))),
                "train_s": float(meta.get("train_total_seconds") or 0.0),
                "latency_ms": float(test_m["inference_ms_per_example"]),
                "throughput": float(test_m["inference_test_examples_per_s"]),
                "best_epoch": int(test_m.get("best_val_f1_epoch") or 0),
                "cm": test_m.get("confusion_matrix"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["trainable_pct"] = 100.0 * df["trainable"] / df["params"].clip(lower=1)
        df["variant"] = df.apply(_variant_label, axis=1)
    return df, missing


def _variant_label(r: pd.Series) -> str:
    if r["mode"] == "unfreeze_last":
        return f"top-{int(r['k'])}/{int(r['L'])}"
    return str(r["method"])


def _short_label(r: pd.Series) -> str:
    label = f"{r['backbone']} {r['variant']}"
    if r["n_train"] > 0:
        label += f" (n={r['n_train']})"
    return label


def _fmt_ppl(v: float) -> str:
    if not math.isfinite(v):
        return "--"
    if v >= 10_000:
        return f"{v:.2e}".replace("e+0", "e").replace("e+", "e")
    return f"{v:.1f}"


def _ranking_figure(df: pd.DataFrame, column: str, xlabel: str, path: Path) -> None:
    d = df.sort_values(column, ascending=True)
    fig, ax = plt.subplots(figsize=(8, 0.24 * len(d) + 1.2))
    colors = [_BACKBONE_COLOR[b] for b in d["backbone"]]
    ax.barh([_short_label(r) for _, r in d.iterrows()], d[column], color=colors)
    for y, v in enumerate(d[column]):
        ax.text(v + 0.3, y, f"{v:.1f}", va="center", fontsize=6)
    lo = max(0.0, float(d[column].min()) - 5)
    ax.set_xlim(lo, float(d[column].max()) + 3)
    ax.set_xlabel(xlabel)
    ax.tick_params(axis="y", labelsize=6)
    handles = [plt.Rectangle((0, 0), 1, 1, color=_BACKBONE_COLOR[b]) for b in _BACKBONE_ORDER]
    ax.legend(handles, _BACKBONE_ORDER, fontsize=6, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _heatmap_figure(grid: pd.DataFrame, path: Path) -> None:
    import numpy as np

    backbones = [b for b in _BACKBONE_ORDER if b in set(grid["backbone"])]
    mat = np.full((len(backbones), len(_METHOD_COLS)), np.nan)
    for _, r in grid.iterrows():
        i = backbones.index(r["backbone"])
        if r["mode"] == "unfreeze_last":
            for f in r["fractions"]:
                mat[i, _METHOD_COLS.index(f"{int(round(f * 100))}%")] = r["acc"]
        else:
            mat[i, _METHOD_COLS.index(r["method"])] = r["acc"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(_METHOD_COLS)))
    ax.set_xticklabels(["full", "LoRA", "sama głowa", "top 10%", "top 20%", "top 30%", "top 50%"], fontsize=8)
    ax.set_yticks(range(len(backbones)))
    ax.set_yticklabels(backbones, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if mat[i, j] < np.nanmean(mat) else "black")
    fig.colorbar(im, ax=ax, label="accuracy test [%]")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _size_figure(grid: pd.DataFrame, path: Path) -> None:
    base = grid[grid["mode"] == "full"].set_index("backbone")["params"]
    grid = grid.assign(params=grid["backbone"].map(base))
    pt = grid[grid["backbone"].str.startswith("pt-")]
    gpt = grid[grid["backbone"] == "GPT-2"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    markers = {"full": "o", "LoRA": "s", "sama głowa": "^"}
    for method, marker in markers.items():
        d = pt[pt["method"] == method].sort_values("params")
        ax.plot(d["params"] / 1e6, d["acc"], marker=marker, label=f"miniLLaMA {method}")
        g = gpt[gpt["method"] == method]
        if not g.empty:
            ax.scatter(g["params"] / 1e6, g["acc"], marker=marker, s=80, color="#e6550d",
                       edgecolor="black", zorder=5)
    best_unf = pt[pt["mode"] == "unfreeze_last"].groupby("backbone").apply(
        lambda d: d.loc[d["acc"].idxmax()], include_groups=False
    ).sort_values("params")
    ax.plot(best_unf["params"] / 1e6, best_unf["acc"], marker="D", linestyle="--",
            label="miniLLaMA najlepsze top-k")
    g = gpt[gpt["mode"] == "unfreeze_last"]
    if not g.empty:
        top = g.loc[g["acc"].idxmax()]
        ax.scatter([top["params"] / 1e6], [top["acc"]], marker="D", s=80, color="#e6550d",
                   edgecolor="black", zorder=5)
    ax.scatter([], [], color="#e6550d", edgecolor="black", label="GPT-2 (ten sam kształt = ta sama metoda)")
    ax.set_xscale("log")
    ax.set_xlabel("liczba parametrów modelu [M] (skala log)")
    ax.set_ylabel("accuracy test [%]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _tradeoff_figure(grid: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    markers = {"full": "o", "LoRA": "s", "sama głowa": "^", "ostatnie bloki": "D"}
    for method, marker in markers.items():
        d = grid[grid["method"] == method]
        colors = [_BACKBONE_COLOR[b] for b in d["backbone"]]
        axes[0].scatter(d["trainable"].clip(lower=1), d["acc"], c=colors, marker=marker,
                        edgecolor="black", linewidth=0.4, s=40, label=method)
        axes[1].scatter(d["train_s"] / 60, d["acc"], c=colors, marker=marker,
                        edgecolor="black", linewidth=0.4, s=40, label=method)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("trenowane parametry (skala log)")
    axes[1].set_xlabel("czas treningu (5 epok) [min]")
    for ax in axes:
        ax.set_ylabel("accuracy test [%]")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, title="kształt = metoda", title_fontsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=_BACKBONE_COLOR[b]) for b in _BACKBONE_ORDER]
    axes[1].legend(handles, _BACKBONE_ORDER, fontsize=7, title="kolor = backbone", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _inference_figure(grid: pd.DataFrame, path: Path) -> None:
    d = grid.sort_values("throughput", ascending=True)
    labels = [_short_label(r) for _, r in d.iterrows()]
    colors = [_BACKBONE_COLOR[b] for b in d["backbone"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.22 * len(d) + 1.2), sharey=True)
    axes[0].barh(labels, d["throughput"], color=colors)
    axes[0].set_xlabel("przepustowość [przykłady/s], batch 32")
    axes[1].barh(labels, d["latency_ms"], color=colors)
    axes[1].set_xlabel("opóźnienie [ms/przykład], batch 1")
    for ax in axes:
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _perplexity_figure(grid: pd.DataFrame, path: Path) -> None:
    import numpy as np

    d = grid[grid["backbone"] != "scratch-5M"].sort_values(["backbone", "variant"])
    d = d.assign(order=d["backbone"].map(_BACKBONE_ORDER.index)).sort_values(["order", "lm_after"])
    labels = [_short_label(r) for _, r in d.iterrows()]
    y = np.arange(len(d))
    fig, axes = plt.subplots(1, 3, figsize=(12, 0.22 * len(d) + 1.4), sharey=True)
    axes[0].barh(y, d["cls_ppl"], color=[_BACKBONE_COLOR[b] for b in d["backbone"]])
    axes[0].axvline(3.0, color="gray", linestyle=":", linewidth=1)
    axes[0].set_xlabel("perplexity klasyfikacji exp(CE)")
    axes[1].scatter(d["lm_before"], y, marker="|", color="black", s=60, label="przed fine-tune")
    axes[1].scatter(d["lm_after"], y, marker="o", color=[_BACKBONE_COLOR[b] for b in d["backbone"]],
                    edgecolor="black", linewidth=0.3, s=25, label="po fine-tune")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("perplexity LM na token (log)")
    axes[1].legend(fontsize=6)
    axes[2].scatter(d["bpc_before"], y, marker="|", color="black", s=60, label="przed")
    axes[2].scatter(d["bpc_after"], y, marker="o", color=[_BACKBONE_COLOR[b] for b in d["backbone"]],
                    edgecolor="black", linewidth=0.3, s=25, label="po")
    axes[2].set_xlabel("bity na znak (LM)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=6)
    for ax in axes:
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _ablation_figure(df: pd.DataFrame, path: Path) -> None:
    keys = [("scratch-5M", "full"), ("pt-110M", "full"), ("GPT-2", "full"), ("GPT-2", "LoRA")]
    full_n = 45_615
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for backbone, method in keys:
        d = df[(df["backbone"] == backbone) & (df["method"] == method)].copy()
        if d.empty:
            continue
        d["n"] = d["n_train"].where(d["n_train"] > 0, full_n)
        d = d.sort_values("n")
        ax.plot(d["n"], d["acc"], marker="o", label=f"{backbone} {method}",
                color=_BACKBONE_COLOR[backbone], linestyle="--" if method == "LoRA" else "-")
    ax.set_xscale("log")
    ax.set_xticks([2000, 10_000, full_n])
    ax.set_xticklabels(["2 000", "10 000", "45 615 (pełny)"])
    ax.minorticks_off()
    ax.set_xlabel("liczba przykładów treningowych")
    ax.set_ylabel("accuracy test [%]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _val_curves_figure(grid: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, method in zip(axes, ("full", "LoRA")):
        for _, r in grid[grid["method"] == method].iterrows():
            hist = pd.read_csv(Path(r["run_dir"]) / METRICS_FILENAME)
            ax.plot(hist["epoch"], 100 * hist["eval_f1_macro"], marker="o", markersize=3,
                    label=r["backbone"], color=_BACKBONE_COLOR[r["backbone"]])
        ax.set_title(f"metoda: {method}", fontsize=9)
        ax.set_xlabel("epoka")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("val F1 macro [%]")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _learning_curves_figure(grid: pd.DataFrame, path: Path) -> None:
    picks = [("scratch-5M", "full"), ("pt-5M", "full"), ("pt-110M", "full"), ("GPT-2", "full"), ("GPT-2", "LoRA")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for backbone, method in picks:
        d = grid[(grid["backbone"] == backbone) & (grid["method"] == method)]
        if d.empty:
            continue
        hist = pd.read_csv(Path(d.iloc[0]["run_dir"]) / METRICS_FILENAME)
        color = _BACKBONE_COLOR[backbone]
        style = "--" if method == "LoRA" else "-"
        label = f"{backbone} {method}"
        axes[0].plot(hist["epoch"], hist["train_loss"], style, marker="o", markersize=3, color=color, label=label)
        axes[1].plot(hist["epoch"], hist["eval_loss"], style, marker="o", markersize=3, color=color, label=label)
        axes[2].plot(hist["epoch"], 100 * hist["train_acc"], style, color=color, alpha=0.45)
        axes[2].plot(hist["epoch"], 100 * hist["eval_acc"], style, marker="o", markersize=3, color=color, label=label)
    axes[0].set_title("strata treningowa (CE)", fontsize=9)
    axes[1].set_title("strata walidacyjna (CE)", fontsize=9)
    axes[2].set_title("accuracy: val (punkty) i train (blado)", fontsize=9)
    for ax in axes:
        ax.set_xlabel("epoka")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
    axes[2].set_ylabel("[%]")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _confusion_figure(grid: pd.DataFrame, path: Path) -> None:
    import numpy as np

    picks = [
        grid[grid["backbone"].str.startswith("pt-")].sort_values("acc").iloc[-1],
        grid[grid["backbone"] == "GPT-2"].sort_values("acc").iloc[-1],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, r in zip(axes, picks):
        cm = np.array(r["cm"], dtype=float)
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{int(cm[i, j])}\n{100 * norm[i, j]:.0f}%", ha="center", va="center",
                        fontsize=7, color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(LABEL_NAMES, fontsize=7)
        ax.set_yticklabels(LABEL_NAMES, fontsize=7)
        ax.set_xlabel("predykcja")
        ax.set_ylabel("prawda")
        ax.set_title(f"{_short_label(r)} (acc {r['acc']:.1f}%)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _results_tables(df: pd.DataFrame, grid: pd.DataFrame) -> None:
    d = grid.sort_values("acc", ascending=False)
    header = (
        r"\# & Backbone & Wariant & Param. & Trenowane & Acc [\%] & F1 macro & F1 ważone & cls PPL "
        r"& Czas [min] & s/epokę \\"
    )
    cls_lines = [
        r"{\footnotesize\setlength{\tabcolsep}{3.5pt}\setlength{\LTcapwidth}{\linewidth}",
        r"\begin{longtable}{@{}rllrrrrrrrr@{}}",
        r"\caption{Wyniki klasyfikacji na teście (12\,284 tweety), posortowane malejąco po accuracy "
        r"(GPT-2 wyróżniony). Param.\ = liczba parametrów modelu [M]; Trenowane = odsetek parametrów "
        r"aktualizowanych przez optymalizator; F1 w \%.}\label{tab:results}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for i, (_, r) in enumerate(d.iterrows(), start=1):
        pct = r["trainable_pct"]
        pct_s = f"{pct:.2f}\\%" if pct < 1 else f"{pct:.0f}\\%"
        row = (
            f"{i} & {r['backbone']} & {r['variant']} & {r['params'] / 1e6:.1f} & {pct_s} & {r['acc']:.2f} & "
            f"{r['f1m']:.2f} & {r['f1w']:.2f} & {r['cls_ppl']:.3f} & {r['train_s'] / 60:.1f} & "
            f"{r['train_s'] / r['epochs']:.0f} \\\\"
        )
        if r["backbone"] == "GPT-2":
            row = r"\rowcolor{orange!12}" + row
        cls_lines.append(row)
    cls_lines += [r"\end{longtable}", r"}"]
    _write(GENERATED / "results_table.tex", "\n".join(cls_lines) + "\n")

    lm = grid[grid["backbone"] != "scratch-5M"].copy()
    lm = lm.assign(order=lm["backbone"].map(_BACKBONE_ORDER.index)).sort_values(["order", "acc"], ascending=[True, False])
    lm_lines = [
        r"{\small\setlength{\tabcolsep}{4pt}\setlength{\LTcapwidth}{\linewidth}",
        r"\begin{longtable}{@{}llrrrrrr@{}}",
        r"\caption{Perplexity modelu językowego na tekstach testowych (przed i po fine-tunie, ten sam backbone "
        r"z wagami \texttt{best\_eval}) oraz szybkość inferencji klasyfikatora. Przepustowość: cały test "
        r"batchami po 32; opóźnienie: 1\,000 pojedynczych przykładów.}\label{tab:lm}\\",
        r"\toprule",
        r"Backbone & Wariant & LM PPL przed & LM PPL po & bpc przed & bpc po & przykł./s & ms/przykł. \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Backbone & Wariant & LM PPL przed & LM PPL po & bpc przed & bpc po & przykł./s & ms/przykł. \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for _, r in lm.iterrows():
        lm_lines.append(
            f"{r['backbone']} & {r['variant']} & {_fmt_ppl(r['lm_before'])} & {_fmt_ppl(r['lm_after'])} & "
            f"{r['bpc_before']:.2f} & {r['bpc_after']:.2f} & {r['throughput']:.0f} & {r['latency_ms']:.1f} \\\\"
        )
    lm_lines += [r"\end{longtable}", r"}"]
    _write(GENERATED / "lm_table.tex", "\n".join(lm_lines) + "\n")

    abl = df[
        df.apply(lambda r: (r["backbone"], r["method"]) in
                 {("scratch-5M", "full"), ("pt-110M", "full"), ("GPT-2", "full"), ("GPT-2", "LoRA")}, axis=1)
    ].copy()
    abl["n"] = abl["n_train"].where(abl["n_train"] > 0, 45_615)
    body = []
    for (backbone, method), g in abl.groupby(["backbone", "method"], sort=False):
        by_n = {int(r["n"]): r for _, r in g.iterrows()}
        cells = []
        for n in (2000, 10_000, 45_615):
            r = by_n.get(n)
            cells.append(f"{r['acc']:.1f} / {r['f1m']:.1f}" if r is not None else "--")
        body.append(f"{backbone} & {method} & " + " & ".join(cells) + r" \\")
    _write(
        GENERATED / "ablation_table.tex",
        _table(
            "Ablacja rozmiaru zbioru treningowego: accuracy / F1 macro na teście [\\%].",
            "tab:ablation",
            "@{}llrrr@{}",
            r"Backbone & Metoda & 2\,000 & 10\,000 & 45\,615 (pełny) \\",
            body,
        ),
    )


def _results_macros(df: pd.DataFrame, grid: pd.DataFrame) -> None:
    def best(mask) -> pd.Series:
        return grid[mask].sort_values("acc").iloc[-1]

    pt = grid["backbone"].str.startswith("pt-")
    gpt = grid["backbone"] == "GPT-2"
    b_all, b_pt, b_gpt = best(pt | gpt | (grid["backbone"] == "scratch-5M")), best(pt), best(gpt)
    scratch = grid[grid["backbone"] == "scratch-5M"].iloc[0]
    pt5_full = grid[(grid["backbone"] == "pt-5M") & (grid["method"] == "full")].iloc[0]
    macros = {
        "nRuns": str(len(df)),
        "bestAllLabel": _short_label(b_all),
        "bestAllAcc": f"{b_all['acc']:.2f}",
        "bestAllFone": f"{b_all['f1m']:.2f}",
        "bestPtLabel": _short_label(b_pt),
        "bestPtAcc": f"{b_pt['acc']:.2f}",
        "bestPtFone": f"{b_pt['f1m']:.2f}",
        "bestGptLabel": _short_label(b_gpt),
        "bestGptAcc": f"{b_gpt['acc']:.2f}",
        "gapPtGpt": f"{b_gpt['acc'] - b_pt['acc']:.1f}",
        "scratchAcc": f"{scratch['acc']:.2f}",
        "ptFiveFullAcc": f"{pt5_full['acc']:.2f}",
        "gainPretrain": f"{pt5_full['acc'] - scratch['acc']:.1f}",
    }
    lines = [rf"\newcommand{{\res{k}}}{{{_tex_escape(v, keep_unicode=True)}}}" for k, v in macros.items()]
    _write(GENERATED / "results_macros.tex", "\n".join(lines) + "\n")


def generate_results_section() -> None:
    df, missing = _collect_results()
    miss_lines = [r"\begin{itemize}"]
    if missing:
        miss_lines += [f"  \\item {_tex_escape(tag)}" for tag in missing]
    else:
        miss_lines.append(r"  \item Wszystkie runy ukończone.")
    miss_lines.append(r"\end{itemize}")
    _write(GENERATED / "missing_runs.tex", "\n".join(miss_lines) + "\n")
    if df.empty:
        _write(
            GENERATED / "results_table.tex",
            r"\noindent\textit{Brak ukończonych runów --- tabela wypełni się po nocnej kolejce.}" + "\n",
        )
        return

    df.drop(columns=["cm", "fractions"]).sort_values("acc", ascending=False).to_csv(
        GENERATED / "results_summary.csv", index=False
    )
    grid = df[df["n_train"] == 0]
    FIGURES.mkdir(parents=True, exist_ok=True)
    _results_tables(df, grid)
    _results_macros(df, grid)
    _ranking_figure(grid, "acc", "accuracy test [%]", FIGURES / "accuracy_ranking.pdf")
    _ranking_figure(grid, "f1m", "F1 macro test [%]", FIGURES / "f1_ranking.pdf")
    _heatmap_figure(grid, FIGURES / "method_heatmap.pdf")
    _size_figure(grid, FIGURES / "acc_vs_size.pdf")
    _tradeoff_figure(grid, FIGURES / "tradeoff.pdf")
    _inference_figure(grid, FIGURES / "inference.pdf")
    _perplexity_figure(grid, FIGURES / "perplexity.pdf")
    _ablation_figure(df, FIGURES / "data_ablation.pdf")
    _val_curves_figure(grid, FIGURES / "val_f1_curves.pdf")
    _learning_curves_figure(grid, FIGURES / "learning_curves.pdf")
    _confusion_figure(grid, FIGURES / "confusion_best.pdf")


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