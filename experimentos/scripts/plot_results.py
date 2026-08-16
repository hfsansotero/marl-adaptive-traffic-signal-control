"""
Genera figuras de resultados para el TFM:
- convergencia_hz4x4.pdf, convergencia_jinan3x4.pdf: ATT (TEST) vs episodio para los 5 algoritmos RL
- comparativa_att.pdf: barras finales por algoritmo x benchmark

Lee los DTL logs de LibSignal en LibSignal/data/output_data/tsc/cityflow_<agent>/<dataset>/exp01/logger/*_DTL.log
Si hay varios logs por (agent, dataset), usa el más reciente.

Output: experimentos/results/figures/
"""
import os
import re
import glob
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(f"{os.path.dirname(__file__)}/../..")
DTL_ROOT = f"{PROJECT_ROOT}/experimentos/LibSignal/data/output_data/tsc"
OUTPUT_DIR = f"{PROJECT_ROOT}/experimentos/results/figures"

DTL_RE = re.compile(
    r"^(\S+)\s+(TRAIN|TEST)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([-\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)"
)

RL_AGENTS = ["presslight", "mplight", "colight", "frap", "axlight"]
ALL_AGENTS = ["fixedtime", "maxpressure"] + RL_AGENTS
DATASETS = ["cityflow4x4", "cityflow_jinan3x4"]

DATASET_LABEL = {
    "cityflow4x4":        "Hangzhou 4$\\times$4",
    "cityflow_jinan3x4":  "Jinan 3$\\times$4",
}

AGENT_LABEL = {
    "fixedtime":    "Fixed Time",
    "maxpressure":  "Max Pressure",
    "presslight":   "PressLight",
    "mplight":      "MPLight",
    "colight":      "CoLight",
    "frap":         "FRAP",
    "axlight":      "AXLight (ATS)",
}

AGENT_COLOR = {
    "fixedtime":    "#666666",
    "maxpressure":  "#999999",
    "presslight":   "#1f77b4",
    "mplight":      "#ff7f0e",
    "colight":      "#2ca02c",
    "frap":         "#d62728",
    "axlight":      "#9467bd",
}

AGENT_STYLE = {
    "presslight":   "-",
    "mplight":      "-",
    "colight":      "-",
    "frap":         "-",
    "axlight":      "-",
}


def find_latest_dtl(agent, dataset):
    pattern = f"{DTL_ROOT}/cityflow_{agent}/{dataset}/exp01/logger/*_DTL.log"
    matches = glob.glob(pattern)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def parse_test_curve(dtl_path):
    """Devuelve dict {episode: att} a partir de filas TEST."""
    curve = {}
    with open(dtl_path) as f:
        for line in f:
            m = DTL_RE.match(line.strip())
            if m:
                _, mode, ep, att, *_ = m.groups()
                if mode == "TEST":
                    curve[int(ep)] = float(att)
    return curve


def get_final_att(agent, dataset):
    """ATT del último TEST disponible (Fixed Time/Max Pressure tienen un único valor)."""
    dtl = find_latest_dtl(agent, dataset)
    if not dtl:
        return None
    curve = parse_test_curve(dtl)
    if not curve:
        return None
    return curve[max(curve.keys())]


def plot_convergence(dataset, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for agent in RL_AGENTS:
        dtl = find_latest_dtl(agent, dataset)
        if not dtl:
            print(f"  [skip] sin log para {agent} × {dataset}")
            continue
        curve = parse_test_curve(dtl)
        if not curve:
            continue
        eps = sorted(curve.keys())
        atts = [curve[e] for e in eps]
        ax.plot(eps, atts,
                label=AGENT_LABEL[agent],
                color=AGENT_COLOR[agent],
                linewidth=1.6,
                alpha=0.9)

    ax.set_xlabel("Episodio")
    ax.set_ylabel("ATT (s) — TEST")
    ax.set_title(f"Convergencia en {DATASET_LABEL[dataset]}")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(0, 200)
    fig.tight_layout()
    fig.savefig(out_path, format="png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_convergence_zoom(dataset, out_path, ylim_max=400):
    """Versión zoom para ver la zona convergida sin que los valores iniciales saturen el eje Y."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for agent in RL_AGENTS:
        dtl = find_latest_dtl(agent, dataset)
        if not dtl:
            continue
        curve = parse_test_curve(dtl)
        if not curve:
            continue
        eps = sorted(curve.keys())
        atts = [curve[e] for e in eps]
        ax.plot(eps, atts,
                label=AGENT_LABEL[agent],
                color=AGENT_COLOR[agent],
                linewidth=1.6,
                alpha=0.9)

    ax.set_xlabel("Episodio")
    ax.set_ylabel("ATT (s) — TEST")
    ax.set_title(f"Convergencia en {DATASET_LABEL[dataset]} (zona estacionaria)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(0, 200)
    ax.set_ylim(280, ylim_max)
    fig.tight_layout()
    fig.savefig(out_path, format="png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_comparison_bars(out_path):
    """Bar chart comparativo: ATT final por algoritmo x dataset."""
    results = {agent: {} for agent in ALL_AGENTS}
    for agent in ALL_AGENTS:
        for dataset in DATASETS:
            att = get_final_att(agent, dataset)
            if att is not None:
                results[agent][dataset] = att

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x_positions = list(range(len(ALL_AGENTS)))
    width = 0.35

    hz_vals    = [results[a].get("cityflow4x4", 0)       for a in ALL_AGENTS]
    jinan_vals = [results[a].get("cityflow_jinan3x4", 0) for a in ALL_AGENTS]

    bars1 = ax.bar([x - width/2 for x in x_positions], hz_vals,    width,
                   label="Hangzhou 4$\\times$4", color="#1f77b4", alpha=0.85)
    bars2 = ax.bar([x + width/2 for x in x_positions], jinan_vals, width,
                   label="Jinan 3$\\times$4",    color="#ff7f0e", alpha=0.85)

    for bar, val in zip(bars1, hz_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, val + 5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, jinan_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, val + 5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([AGENT_LABEL[a] for a in ALL_AGENTS], rotation=20, ha="right")
    ax.set_ylabel("ATT final (s)")
    ax.set_title("Comparativa de ATT final por algoritmo y benchmark")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, 650)
    fig.tight_layout()
    fig.savefig(out_path, format="png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  -> {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Curvas de convergencia (rango completo):")
    plot_convergence("cityflow4x4",       f"{OUTPUT_DIR}/convergencia_hz4x4.png")
    plot_convergence("cityflow_jinan3x4", f"{OUTPUT_DIR}/convergencia_jinan3x4.png")

    print("\nCurvas de convergencia (zoom en zona estacionaria):")
    plot_convergence_zoom("cityflow4x4",       f"{OUTPUT_DIR}/convergencia_hz4x4_zoom.png",   ylim_max=400)
    plot_convergence_zoom("cityflow_jinan3x4", f"{OUTPUT_DIR}/convergencia_jinan3x4_zoom.png", ylim_max=380)

    print("\nComparativa de barras:")
    plot_comparison_bars(f"{OUTPUT_DIR}/comparativa_att.png")

    print("\nListo.")


if __name__ == "__main__":
    main()
