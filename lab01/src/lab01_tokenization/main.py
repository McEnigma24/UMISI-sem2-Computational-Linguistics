"""CLI entry point for lab01 tokenization benchmark."""

from __future__ import annotations

import argparse
import os
import shlex
import sys

from lab01_tokenization.config import TOKENIZER_NAMES


def _slurm_cpu_summary() -> str:
    ntasks = os.environ.get("SLURM_NTASKS", "?")
    cpt = os.environ.get("SLURM_CPUS_PER_TASK", "?")
    omp = os.environ.get("OMP_NUM_THREADS", "?")
    try:
        total = int(ntasks) * int(cpt)
    except ValueError:
        total = "?"
    return (
        f"SLURM_NTASKS={ntasks} SLURM_CPUS_PER_TASK={cpt} "
        f"OMP_NUM_THREADS={omp} (łącznie rdzeni CPU ≈ {total})"
    )


def _add_tokenizer_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--tokenizer",
        choices=TOKENIZER_NAMES,
        required=True,
        help="which tokenizer pipeline to use",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lab01", description="Lab01 tokenization benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare-data", help="download WikiText-103 raw splits")

    p_tok = sub.add_parser("train-tokenizers", help="train/build all three tokenizers")
    p_tok.add_argument(
        "--force",
        action="store_true",
        help="rebuild even if artifacts exist (refreshes train_wall_seconds)",
    )

    p_enc = sub.add_parser("encode", help="encode splits to uint16 bins")
    p_enc.add_argument("--tokenizer", choices=TOKENIZER_NAMES, default=None)
    p_enc.add_argument("--max-train-mb", type=float, default=None)

    p_stats = sub.add_parser("tok-stats", help="tokenization statistics on test")
    p_stats.add_argument("--tokenizer", choices=TOKENIZER_NAMES, default=None)

    sub.add_parser("qualitative", help="write qualitative tokenization markdown")

    p_train = sub.add_parser("train", help="train LM for one tokenizer")
    _add_tokenizer_arg(p_train)
    p_train.add_argument("--smoke", action="store_true")
    p_train.add_argument("--max-train-mb", type=float, default=None)

    p_eval = sub.add_parser("eval", help="evaluate LM perplexity")
    _add_tokenizer_arg(p_eval)

    sub.add_parser("report", help="aggregate metrics and update REPORT.md")

    p_all = sub.add_parser("all", help="full pipeline (optional smoke)")
    p_all.add_argument("--smoke", action="store_true")
    p_all.add_argument("--max-train-mb", type=float, default=None)
    p_all.add_argument("--skip-train", action="store_true", help="data/tokenizer steps only")

    return parser


def run_from_argv(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv and os.environ.get("LAB01_CMD"):
        argv = shlex.split(os.environ["LAB01_CMD"])
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare-data":
        from lab01_tokenization.data import prepare_data

        prepare_data()
    elif args.command == "train-tokenizers":
        from lab01_tokenization.stats import save_tokenizer_train_times
        from lab01_tokenization.tokenizers_impl import train_all_tokenizers

        times = train_all_tokenizers(force=args.force)
        save_tokenizer_train_times(times)
        print("tokenizer train times:", times)
    elif args.command == "encode":
        from lab01_tokenization.encode import encode_all

        encode_all(args.tokenizer, max_train_mb=args.max_train_mb)
    elif args.command == "tok-stats":
        from lab01_tokenization.stats import compute_stats

        compute_stats(args.tokenizer)
    elif args.command == "qualitative":
        from lab01_tokenization.qualitative import run_qualitative

        run_qualitative()
    elif args.command == "train":
        from lab01_tokenization.train import train_tokenizer_lm

        train_tokenizer_lm(
            args.tokenizer,
            smoke=args.smoke,
            max_train_mb=args.max_train_mb,
        )
    elif args.command == "eval":
        from lab01_tokenization.evaluate import evaluate_tokenizer

        evaluate_tokenizer(args.tokenizer)
    elif args.command == "report":
        from lab01_tokenization.report import run_report

        run_report()
    elif args.command == "all":
        from lab01_tokenization.data import prepare_data
        from lab01_tokenization.encode import encode_all
        from lab01_tokenization.evaluate import evaluate_tokenizer
        from lab01_tokenization.qualitative import run_qualitative
        from lab01_tokenization.report import run_report
        from lab01_tokenization.stats import compute_stats, save_tokenizer_train_times
        from lab01_tokenization.tokenizers_impl import train_all_tokenizers
        from lab01_tokenization.train import train_tokenizer_lm

        prepare_data()
        if not args.skip_train:
            save_tokenizer_train_times(train_all_tokenizers())
        encode_all(max_train_mb=args.max_train_mb)
        compute_stats()
        run_qualitative()
        if not args.skip_train:
            for name in TOKENIZER_NAMES:
                train_tokenizer_lm(name, smoke=args.smoke, max_train_mb=args.max_train_mb)
                evaluate_tokenizer(name)
        run_report()
    else:
        parser.error(f"unknown command {args.command}")


def main() -> None:
    if "LOCAL_RANK" in os.environ and os.environ.get("LAB01_TORCHRUN_SMOKE") == "1":
        import torch

        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        print(
            f"torchrun local_rank={local_rank} "
            f"world_size={os.environ.get('WORLD_SIZE')} "
            f"device={torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'}"
        )
        if local_rank == 0:
            print(_slurm_cpu_summary())
        return

    run_from_argv()


if __name__ == "__main__":
    main()
