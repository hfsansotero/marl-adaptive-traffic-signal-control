"""
Genera mapas esquemáticos de las redes de tráfico hz4x4 y jinan3x4 a partir de los
ficheros roadnet JSON de CityFlow. Dibuja intersecciones, carreteras y etiqueta cada
intersección con su índice para que se pueda referenciar en el texto.

Output: experimentos/results/figures/red_hz4x4.png, red_jinan3x4.png
"""
import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PROJECT_ROOT = os.path.abspath(f"{os.path.dirname(__file__)}/../..")
RAW_ROOT     = f"{PROJECT_ROOT}/experimentos/LibSignal/data/raw_data"
OUTPUT_DIR   = f"{PROJECT_ROOT}/experimentos/results/figures"

NETWORKS = {
    "hz4x4": {
        "roadnet":   f"{RAW_ROOT}/hangzhou_4x4_gudang_18041610_1h/roadnet_4X4.json",
        "title":     "Hangzhou 4$\\times$4 (16 intersecciones)",
        "out":       f"{OUTPUT_DIR}/red_hz4x4.png",
    },
    "jinan3x4": {
        "roadnet":   f"{RAW_ROOT}/jinan_3x4/roadnet_3_4.json",
        "title":     "Jinan 3$\\times$4 (12 intersecciones)",
        "out":       f"{OUTPUT_DIR}/red_jinan3x4.png",
    },
}


def plot_network(cfg):
    with open(cfg["roadnet"]) as f:
        rn = json.load(f)

    intersections = rn["intersections"]
    roads         = rn["roads"]

    road_by_id = {r["id"]: r for r in roads}

    fig, ax = plt.subplots(figsize=(8, 8))

    for r in roads:
        pts = r["points"]
        if len(pts) < 2:
            continue
        xs = [p["x"] for p in pts]
        ys = [p["y"] for p in pts]
        ax.plot(xs, ys, color="#888888", linewidth=1.0, alpha=0.7, zorder=1)

    signal_inters = [i for i in intersections if not i.get("virtual", False)]
    virtual_inters = [i for i in intersections if i.get("virtual", False)]

    for i in virtual_inters:
        ax.scatter(i["point"]["x"], i["point"]["y"],
                   s=18, marker="o", color="#cccccc",
                   edgecolors="#999999", linewidth=0.5, zorder=2)

    for idx, i in enumerate(signal_inters):
        x, y = i["point"]["x"], i["point"]["y"]
        ax.scatter(x, y, s=180, marker="s",
                   color="#1f77b4", edgecolors="black", linewidth=0.8, zorder=3)
        ax.text(x, y, str(idx),
                ha="center", va="center", fontsize=8, color="white",
                fontweight="bold", zorder=4)

    ax.set_aspect("equal")
    ax.set_title(cfg["title"])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, linestyle="--", alpha=0.3)

    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w',
                   markerfacecolor='#1f77b4', markeredgecolor='black',
                   markersize=10, label='Intersección con semáforo'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#cccccc', markeredgecolor='#999999',
                   markersize=6, label='Nodo virtual (entrada/salida)'),
        plt.Line2D([0], [0], color='#888888', linewidth=1.2,
                   label='Carretera'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9, fontsize=9)

    fig.tight_layout()
    fig.savefig(cfg["out"], format="png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  -> {cfg['out']}  ({len(signal_inters)} intersecciones con semáforo, "
          f"{len(virtual_inters)} virtuales, {len(roads)} carreteras)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for key, cfg in NETWORKS.items():
        print(f"Generando red {key}:")
        plot_network(cfg)


if __name__ == "__main__":
    main()
