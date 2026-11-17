"""Domyślny punkt wejścia w kontenerze: run_main"""

from __future__ import annotations

import os
import subprocess

import infer
import train
from _utils_.log_print import log_print, reset_log_timer, tensorboard_run_dir

from training_presets import DUAL3060_ALL

def main() -> str:
    reset_log_timer()
    log_root = os.environ.get("TENSORBOARD_LOGDIR", "/workspace/runs")
    run_dir = tensorboard_run_dir(log_root)

    try:
        import torch

        log_print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
        log_print(f"TensorBoard logi: {run_dir}  (w shellu: tb)")
    except ImportError:
        log_print("Brak torch — uruchom na hoście: ./scripts/drun sync")

    subprocess.run(["clear"], check=False)
    return run_dir

if __name__ == "__main__":
    run_dir = main()


    # DATASET load #
    if os.environ.get("LANGUAGE_RNN_MAX_CORPUS_CHARS"):
        log_print(
            f"Limit korpusu: {int(os.environ['LANGUAGE_RNN_MAX_CORPUS_CHARS']):,} znaków "
            f"(usuń env dla pełnego datasetu)"
        )
    else:
        log_print("Korpus: pełny dataset (bez LANGUAGE_RNN_MAX_CORPUS_CHARS)")



    # TRAIN THEM ALL #
    for architecture, preset in DUAL3060_ALL:
        # train.main(tensorboard_dir=run_dir, architecture=architecture, resume="latest", **preset)
        train.main(tensorboard_dir=run_dir, architecture=architecture, **preset)



    # # COMPARE THEM ALL #
    # length = 25
    # infer.compare_models(
    #     models=[(arch, None) for arch, _ in DUAL3060_ALL],
    #     prompts=[
    #         ("AGH to ", length),
    #         ("Język polski do cudowne dzieło ludzkości ", length),
    #         ("Naprawę ? ", length),
    #         ("Dobrze że mówisz, prawie bym zapomniał ", length),


    #         ("Obelixie zostaw ten menhir ", length),
    #         ("na teutatesa ", length),


    #         ("Smerfetka jest bezpieczna, została w ", length),


    #         ("Uważaj wbijają na B, idź ", length),
    #         ("Plantuje pakę, masz 30 sekund na ", length),


    #         ("postęp w AI jest poje", length),


    #         ("Mogę ci powiedzeć ale musiałbym cię potem ", length),


    #         ("Geralt, a ty co powiesz? Jak ci sandaczyk smakuje? ", length),
    #         ("Obudził się w tobie poeta? - Mhm. Chcesz usłyszeć fraszkę? - Pewnie - Lambert, Lambert, ty chuju ", length),
    #         ("Na co się tak patrzysz ? ", length),



    #         ("ale stara ta mapa uważaj bo się po ", length),


    #         ("Sensem życia jest ", length),
    #     ],
    #     temperature=0.85,
    #     run_tag="dual3060-110M",
    # )







# seed = "Na zamku"
# length = 25

# infer.main(architecture="rnnRNN", seed=seed,
#             checkpoint_path="saved_models/rnnRNN-2026.11.01-011843-quick-test/rnnRNN_final.pt",
#             length=length,
# )

# infer.main(architecture="rnnLSTM", seed=seed,
#             checkpoint_path="saved_models/rnnLSTM-2026.11.01-012810-quick-test/rnnLSTM_final.pt",
#             length=length,
# )

# infer.main(architecture="rnnGRU", seed=seed,
#             checkpoint_path="saved_models/rnnGRU-2026.11.01-014552-quick-test/rnnGRU_final.pt",
#             length=length,
# )



# infer.compare_models(
#     models=[
#         ("rnnRNN", "saved_models/rnnRNN-2026.11.01-011843-quick-test/rnnRNN_final.pt"),
#         ("rnnLSTM", "saved_models/rnnLSTM-2026.11.01-012810-quick-test/rnnLSTM_final.pt"),
#         ("rnnGRU", "saved_models/rnnGRU-2026.11.01-014552-quick-test/rnnGRU_final.pt"),
#     ],
#     prompts=[(seed, length), ("W polskim", 50)],
#     temperature=0.85,
#     run_tag="manual-compare",
# )