# MARL for intelligent traffic signals

**Language / Idioma**: [ES](README.md) | **EN**

Experimental Master’s Thesis pipeline on **adaptive urban traffic signal control with Multi-Agent Reinforcement Learning (MARL)**, evaluating seven algorithms on two real traffic benchmarks (Hangzhou 4×4 and Jinan 3×4) using the CityFlow simulator and the LibSignal framework.

- **Author**: Héctor Fernández San Sotero
- **Supervisor**: Francisco Soltero
- **Institution**: Universidad Alfonso X el Sabio (UAX)
- **Academic year**: 2025-2026

> Bilingual maintenance policy: `README.md` is the primary Spanish version. The English version lives in `README.en.md`; any content change in one file must be replicated in the other to keep both versions aligned.

---

## What is this project about?

Traditional traffic signals operate with predefined cycles that do not adapt to real traffic demand. MARL-based approaches allow each intersection to act as an agent that learns its signaling policy while cooperating with neighbors, reducing travel times by 25% to 44% in simulation.

Using **LibSignal** (which unifies several state-of-the-art simulators and algorithms), seven algorithms were evaluated—two classic baselines and five representative MARL methods from the three main state representation families—on the two most widely used reference benchmarks in the literature.

### Research question

> *How does state representation affect the performance of MARL algorithms for adaptive urban traffic signal control?*

### Original contributions

1. **PyTorch port of Advanced-XLight** (Zhang et al., ICML 2022) and integration as a native LibSignal agent. The original implementation is in TensorFlow 2.4 and was not compatible with LibSignal’s API (see `LibSignal/agent/axlight.py`).
2. **GPU patches for LibSignal**: the public LibSignal repository has a documented issue (#35) that prevents proper GPU use in PressLight, CoLight, and MPLight. Four agents and the trainer were patched, reducing training time by 67% to 86% for lightweight architectures.
3. **Reproducible experimental pipeline**: sequential queue launcher scripts, real-time progress monitor, automatic figure generation, and fairness analysis using the Gini index over per-intersection queue length.
4. **Comparative analysis of the three state representations**—queue, pressure, and ATS—under identical experimental conditions, also evaluating spatial fairness.

The full thesis document (theoretical background, detailed methodology, and discussion of results) is delivered separately as a PDF and is not part of this repository.

---

## Demos: adaptive control in action

Replays extracted with the official CityFlow visualizer from real experiment replay logs, showing learned policy behavior against the fixed-time baseline.

| Fixed Time — Hangzhou 4×4 | Advanced-XLight — Hangzhou 4×4 |
| :---: | :---: |
| ![Fixed Time in Hangzhou 4x4](experimentos/results/demos/fixedtime_hz4x4.gif) | ![AXLight in Hangzhou 4x4](experimentos/results/demos/axlight_hz4x4.gif) |
| Fixed cycles with no adaptation to real demand | Original PyTorch port, contribution of this work (§ below) |

| CoLight — Jinan 3×4 |
| :---: |
| ![CoLight in Jinan 3x4](experimentos/results/demos/colight_jinan3x4.gif) |
| Best ATT and most fair algorithm on this benchmark (explicit coordination via GAT) |

---

## Repository structure

```text
tfm-sem/
├── README.md                            ← primary Spanish README
├── README.en.md                         ← this file
│
└── experimentos/
    ├── LibSignal/                       ← LibSignal repo with GPU patches
    │   ├── agent/                       ← implementations of each algorithm
    │   │   ├── axlight.py               ← original contribution (PyTorch port)
    │   │   ├── frap.py                  ← GPU patched
    │   │   ├── presslight.py            ← GPU patched
    │   │   ├── colight.py               ← GPU patched
    │   │   └── mplight.py               ← GPU patched (via PFRL)
    │   ├── configs/tsc/                 ← YAML configurations by algorithm
    │   ├── data/
    │   │   ├── raw_data/                ← traffic datasets (Jinan, Hangzhou, and others)
    │   │   └── output_data/             ← logs and models generated during training
    │   │       └── tsc/cityflow_<agent>/<dataset>/exp01/
    │   │           ├── logger/          ← *_DTL.log (metrics) and *_BRF.log (per intersection)
    │   │           └── model/           ← checkpoints
    │   └── run.py                       ← single-experiment launcher
    │
    ├── scripts/                         ← pipeline helper scripts
    │   ├── run_experiments.sh           ← sequential experiment queue
    │   ├── queue.txt                    ← list of runs to execute
    │   ├── monitor.py                   ← real-time progress monitor
    │   ├── gini_analysis.py             ← fairness analysis by intersection
    │   ├── extract_metrics.py           ← exports final metrics to CSV
    │   ├── plot_results.py              ← generates convergence figures and bar chart
    │   └── plot_roadnets.py             ← generates road network maps
    │
    └── results/                         ← official thesis logs and results
        ├── log_<agent>_<dataset>.txt    ← stdout from each experiment
        ├── gini_summary.csv             ← computed Gini index
        ├── figures/                     ← figures generated by plot_results.py / plot_roadnets.py
        └── demos/                       ← visual replay GIFs (see "Demos" section above)
```

### Where are the key things?

| What I need                                           | Where it is                                                                                 |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Traffic datasets (CityFlow)                           | `experimentos/LibSignal/data/raw_data/`                                                    |
| Final output logs for each experiment                 | `experimentos/results/log_<agente>_<dataset>.txt` (stdout)                                  |
| Episode-by-episode metrics (TRAIN and TEST)           | `experimentos/LibSignal/data/output_data/tsc/cityflow_<agente>/<dataset>/exp01/logger/*_DTL.log` |
| Metrics broken down by intersection (last TEST ep)    | `experimentos/LibSignal/data/output_data/.../logger/*_BRF.log`                             |
| Trained models (PyTorch checkpoints)                  | `experimentos/LibSignal/data/output_data/.../model/`                                       |
| AXLight port implementation                           | `experimentos/LibSignal/agent/axlight.py`                                                  |
| YAML configuration for each algorithm                 | `experimentos/LibSignal/configs/tsc/<agente>.yml`                                          |
| Fairness summary (CSV)                                | `experimentos/results/gini_summary.csv`                                                    |
| Generated figures (convergence curves, maps)          | `experimentos/results/figures/`                                                            |

---

## Reproducing experiments

### Environment setup

```bash
# 1. Create conda environment
conda create -n tfm python=3.9 -y
conda activate tfm

# 2. PyTorch + CUDA 12.1 (compatible with CUDA 13.2 driver)
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# 3. PyTorch Geometric (required for CoLight)
pip install torch_geometric
pip install torch_scatter torch_sparse \
    -f https://data.pyg.org/whl/torch-2.1.2+cu121.html

# 4. LibSignal dependencies
cd experimentos/LibSignal
pip install -r requirements.txt

# 5. CityFlow (build/install from source)
cd ../cityflow
pip install .

# 6. matplotlib (figures and analysis)
pip install matplotlib==3.9.4
```

### Run a single experiment

From `experimentos/LibSignal/`:

```bash
python run.py -a axlight -w cityflow -n cityflow4x4 --ngpu 0 --prefix exp01
```

Parameters:

- `-a` agent: `fixedtime`, `maxpressure`, `presslight`, `mplight`, `colight`, `frap`, `axlight`
- `-n` dataset: `cityflow4x4` (Hangzhou 4×4) or `cityflow_jinan3x4` (Jinan 3×4)
- `--ngpu 0` uses GPU 0; `--ngpu -1` forces CPU
- `--prefix` batch name (changing it generates a different output directory)

### Run the full queue

From `experimentos/scripts/`:

```bash
bash run_experiments.sh --queue queue.txt --gpu 0 --prefix exp01
```

The script reads `queue.txt` (format: `<agente> <dataset>` per line; lines starting with `#` are ignored) and launches each experiment in sequence, writing stdout to `experimentos/results/log_<agente>_<dataset>.txt`.

### Monitor progress

From `experimentos/` (while runs are active):

```bash
python scripts/monitor.py --logdir LibSignal/data/output_data
```

Displays current and best TEST ATT, estimated ETA, training evolution, and active/paused status for each run.

### Reproduce post-experiment analysis

Once experiments complete:

```bash
# Fairness analysis: Gini index over mean_queue by intersection
python scripts/gini_analysis.py

# Extract final metrics (ATT, AQL, delay, throughput) to CSV
python scripts/extract_metrics.py

# Convergence figures and comparison bar chart (output in experimentos/results/figures/)
python scripts/plot_results.py

# Road network maps for hz4x4 and jinan3x4 (output in experimentos/results/figures/)
python scripts/plot_roadnets.py
```

---

## Script summary

| Script                                | What it does                                                                                | Output                                                  |
| ------------------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| `run_experiments.sh`                  | Reads `queue.txt` and launches experiments sequentially                                     | Logs in `results/log_<a>_<d>.txt` + DTL/BRF in LibSignal |
| `queue.txt`                           | List of runs to execute (`<agente> <dataset>` per line)                                     | —                                                       |
| `monitor.py`                          | Reads active DTL logs and shows SB3-style progress                                          | Interactive stdout                                      |
| `gini_analysis.py`                    | Computes Gini index on `mean_queue` by intersection from BRF logs                           | `results/gini_summary.csv`                              |
| `extract_metrics.py`                  | Extracts final ATT/AQL/delay/throughput for all experiments                                 | `results/metrics_summary.csv`                           |
| `plot_results.py`                     | Generates convergence curves (full range + zoom) and comparative bar chart                  | 5 PNGs in `results/figures/`                            |
| `plot_roadnets.py`                    | Reads CityFlow `roadnet.json` files and draws schematic network maps                        | 2 PNGs in `results/figures/`                            |

---

## Summary of results

| Algorithm       | Representation | Hangzhou 4×4 ATT | Jinan 3×4 ATT | Gini hz4x4 | Gini jinan |
| --------------- | -------------- | ----------------: | -------------: | ---------: | ---------: |
| Fixed Time      | —              |           575.6 s |        425.3 s |          — |          — |
| Max Pressure    | Pressure       |           365.1 s |        329.0 s |          — |          — |
| PressLight      | Pressure       |           330.6 s |        304.8 s |      0.291 |      0.113 |
| MPLight         | Pressure       |           331.3 s |        309.2 s |      0.415 |      0.204 |
| CoLight         | Queue          |           337.1 s |    **302.2 s** |  **0.236** |  **0.090** |
| FRAP            | Queue          |           327.8 s |        305.1 s |      0.418 |      0.179 |
| Advanced-XLight | ATS            |       **323.3 s** |        304.3 s |      0.358 |      0.229 |

**Main findings** (valid under the evaluated experimental conditions: the two benchmarks above, single seed, LibSignal default hyperparameters):

- AXLight achieves the best ATT in Hangzhou 4×4, but CoLight is best in Jinan 3×4 and notably fairer in both.
- The three state representations converge to a similar ceiling in Jinan 3×4 (7 s band); the difference is larger in Hangzhou 4×4 (14 s band between queue and ATS), where the larger network gives more room for enriched representations.
- There is a **clear trade-off between global efficiency (ATT) and spatial fairness (Gini)**: AXLight prioritizes efficiency, CoLight prioritizes fairness.

---

## Code notes and known limitations

- **Single seed**. All experiments were run with `seed=0` due to compute constraints (~98 total GPU hours). This is the main methodological limitation: results are a controlled experimental comparison under identical protocol, not a statistical estimate of mean performance.
- **Chinese datasets**. Benchmarks come from traffic camera data in Hangzhou and Jinan. Generalizing to irregular European topologies requires additional benchmarks (RESCO on SUMO is a natural next step).
- **GPU is not the bottleneck**. In small networks (16 intersections), CPU-saturated CityFlow limits throughput for faster algorithms (PressLight, MPLight)—GPU is idle most of the time.
- **Stdout `.txt` logs**. These are informative console captures during training. Real analysis metrics are in native LibSignal logs (`*_DTL.log` and `*_BRF.log`).
