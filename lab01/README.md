# Lab01 — tokenization

## CLI workflow (default)

Dataset: **WikiText-103 raw** (`Salesforce/wikitext`). Three tokenizers at **N = 50257**: GPT-2 BPE, custom whitespace+`<UNK>`, SentencePiece Unigram.

```bash
cd lab01
source local/env.sh   # optional: HF_HOME, UV_CACHE on $SCRATCH
uv sync

# Login (network for first download)
uv run lab01 prepare-data

# CPU batch: tokenizers + encode + stats (no LM training)
./grid/run_cpu.sh 1 4
# or: LAB01_CMD="all --skip-train" ./grid/run_cpu.sh 1 4

# GPU smoke test
LAB01_CMD="all --smoke" ./grid/run_gpu.sh 1 4

# One tokenizer (train + eval in one GPU job)
LAB01_SHELL_CMD='uv run lab01 train --tokenizer gpt2 && uv run lab01 eval --tokenizer gpt2' ./grid/run_gpu.sh 1 4

# Full pipeline (long)
./grid/run_gpu.sh 1 4

uv run lab01 report
```

Commands: `prepare-data`, `train-tokenizers`, `encode`, `tok-stats`, `qualitative`, `train --tokenizer {gpt2,whitespace,spm}`, `eval --tokenizer …`, `report`, `all [--smoke] [--skip-train]`.

Artifacts: `data/tokenizers/`, `data/bins/<tok>/`, `results/metrics_*.json`, `results/qualitative.md`, [`REPORT.md`](REPORT.md).

Grid jobs read **`LAB01_CMD`** (shell words passed to `uv run lab01`). Set `HF_HUB_OFFLINE=1` on compute nodes after caches are warm.

## Jupyter w Cursorze (Remote SSH)

1. Połącz się z Ares przez **Remote SSH** i otwórz folder **`lab01`** jako workspace (nie cały `$HOME`).
2. Jednorazowo:

   ```bash
   ./local/jupyter_setup.sh
   ```

3. **Reload Window** w Cursorze (`Ctrl+Shift+P` → „Developer: Reload Window”).
4. Otwórz [`notebooks/00_smoke.ipynb`](notebooks/00_smoke.ipynb).

### Kernel — co wybrać w UI

W prawym górnym rogu notebooka **Select Kernel**:

| Opcja | Kiedy |
|--------|--------|
| **Python Environments** → `.venv (Python 3.x)` → ścieżka `.../lab01/.venv/bin/python` | najprostsze; to samo co poniżej |
| **Jupyter Kernel** → **lab01 (uv)** | po `./local/jupyter_setup.sh` |

Nie wybieraj `/usr/bin/python3.9` ani Python 2.7 — to systemowe interpretery loginu.

### Automatyczny wybór (w repo)

Plik [`.vscode/settings.json`](.vscode/settings.json) ustawia:

- `python.defaultInterpreterPath` → `${workspaceFolder}/.venv/bin/python`
- ukrywa systemowe Pythony w liście Jupyter

Działa, gdy workspace = katalog `lab01`. Commituj `.vscode/` do gita — cała grupa ma ten sam default.

Ręcznie (jeśli lista pusta): `Ctrl+Shift+P` → **Python: Select Interpreter** → **Enter interpreter path** →

```text
/net/afscra/people/plgwmackowiak/UMISI-sem2-Computational-Linguistics/lab01/.venv/bin/python
```

(adaptuj do swojej ścieżki w `$SCRATCH`).

Środowisko = ten sam `.venv` co `uv run python -m lab01_tokenization`.

Kernel na **loginie** — tylko lekka eksploracja. Cięższe rzeczy → grid lub Jupyter na węźle (poniżej).

## Jupyter na węźle SLURM (opcja 2)

Interaktywna alokacja (domyślnie **2 h**, `export GRID_JUPYTER_TIME=01:30:00` aby zmienić):

```bash
./grid/run_cpu.sh jupy 2 2       # CPU (skróty: j, jupy, jupyter, jp)
./grid/run_gpu.sh j 1 1          # GPU
```

Po wejściu na węzeł:

```bash
./local/start_jupyter.sh
cat grid/.jupyter_connect    # jedna linia URL — wklej w Cursor
```

W Cursorze (Remote SSH = **login**): **Select Kernel → Existing Jupyter Server** → wklej **cały** URL z `grid/.jupyter_connect`.

Domyślnie Jupyter nasłuchuje na `0.0.0.0` w jobie SLURM, więc URL wygląda jak  
`http://ag0003:8888/lab?token=…` — **nie** `127.0.0.1` (to byłby login → ECONNREFUSED).

Jeśli `ag0003:8888` nie przechodzi z loginu, na węźle:

```bash
export JUPYTER_IP=127.0.0.1
./local/start_jupyter.sh
```

Na **loginie** w drugim terminalu:

```bash
./local/jupyter_tunnel.sh ag0003 8888
```

Wtedy w Cursorze URL z `127.0.0.1` z `grid/.jupyter_connect`.

## Grid (batch)

```bash
./grid/run_cpu.sh [NTASKS] [CPUS_PER_TASK]
./grid/run_gpu.sh [GPUS] [CPUS_PER_TASK]
```

Smoke (login, CPU): `uv run lab01 all --smoke --max-train-mb 2`.
