# Szablon labu PLGrid (Ares) + `uv`

Gotowy projekt: **grid** (batch + Jupyter na węźle), **src/plgrid_lab**, **torch** (CUDA 12.6 / V100), **MPI**, Jupyter w Cursorze.

## Nowy lab

```bash
cp -a _project_template_PLgrid_python_ lab02
cd lab02
# opcjonalnie: zmień name w pyproject.toml i przenieś src/plgrid_lab → src/lab02_*
```

Na Ares kopiuj folder z dysku (`cp -a`) — wtedy masz gotowe `.venv`. Po `git clone` (w repo jest `*venv*` w `.gitignore`) uruchom: `uv sync --group dev`.

### Checklist po skopiowaniu do `labNN`

| Co | Gdzie |
|----|--------|
| `name`, `[project.scripts]` | `pyproject.toml` |
| pakiet Python | `src/plgrid_lab` → `src/labNN_*` + importy |
| `python -m …` | `grid/job_cpu`, `grid/job_gpu` |
| kernel Jupyter | `local/jupyter_setup.sh` (`PLGRID_KERNEL_NAME`) |

## Grid

```bash
./grid/run_cpu.sh 2 1              # batch smoke (MPI)
./grid/run_gpu.sh 1 4              # batch GPU
./grid/run_cpu.sh j 1 4            # Jupyter na węźle CPU (jupy, jupyter, jp)
./grid/run_gpu.sh jupy 1 8         # Jupyter na węźle GPU
./grid/status.sh
./grid/cancel.sh
```

Na węźle po `j`: `./local/start_jupyter.sh` → `cat grid/.jupyter_connect` → Cursor **Existing Jupyter Server**.

## Cursor (login)

```bash
./local/jupyter_setup.sh   # kernel „plgrid-template (uv)” — tylko lekkie testy
```

Otwórz folder projektu jako workspace → `.vscode/settings.json` wskazuje `.venv`.

## Smoke

```bash
uv run python -m plgrid_lab
# lub po sync: ./grid/run_cpu.sh 1 1
```

## Zależności

W `pyproject.toml`: mpi4py, openmpi, torch (index cu126), numpy, transformers, datasets, tqdm; grupa `dev`: jupyterlab, ipykernel.

Nie uruchamiaj ciężkich jobów na **loginie** — tylko `grid/run_*.sh` lub `srun` przez tryb `j`.
