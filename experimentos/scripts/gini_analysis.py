"""
Calcula el índice de Gini sobre mean_queue por intersección en el último
episodio TEST de cada algoritmo x dataset. Lee los BRF logs de LibSignal.

Uso:
    python gini_analysis.py
"""
import os
import re
import glob

BRF_ROOT = "/home/hector/tfm-sem/experimentos/LibSignal/data/output_data/tsc"
OUTPUT_CSV = "/home/hector/tfm-sem/experimentos/results/gini_summary.csv"

INTERSECTION_RE = re.compile(r"intersection:(\d+),\s*mean_episode_reward:[-\d.]+,\s*mean_queue:([\d.]+)")
EPISODE_RE = re.compile(r"episode:(\d+)/(\d+),\s*real avg travel time:([\d.]+)")


def gini(values):
    """Índice de Gini estándar. Entrada: lista de valores no-negativos."""
    if not values or sum(values) == 0:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    cumulative = 0.0
    for i, v in enumerate(sorted_v, 1):
        cumulative += (2 * i - n - 1) * v
    return cumulative / (n * sum(sorted_v))


def parse_last_test_episode(brf_path):
    """Devuelve la lista de mean_queue por intersección del último bloque TEST."""
    with open(brf_path) as f:
        lines = f.readlines()

    last_block_start = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("episode:199/200"):
            last_block_start = i
            break

    if last_block_start is None:
        return None

    queues = []
    for line in lines[last_block_start + 1:]:
        m = INTERSECTION_RE.match(line.strip())
        if m:
            queues.append(float(m.group(2)))
        elif line.startswith("Test step:") or line.startswith("Final Travel Time"):
            break

    return queues


def find_brf_logs():
    """Busca BRF logs estructurados como cityflow_<agent>/<dataset>/exp01/logger/*BRF.log"""
    pattern = os.path.join(BRF_ROOT, "cityflow_*", "*", "exp01", "logger", "*_BRF.log")
    return glob.glob(pattern)


def main():
    logs = find_brf_logs()
    results = []

    by_key = {}
    for path in logs:
        parts = path.replace("\\", "/").split("/")
        idx = next(i for i, p in enumerate(parts) if p.startswith("cityflow_"))
        agent = parts[idx].replace("cityflow_", "")
        dataset = parts[idx + 1]
        mtime = os.path.getmtime(path)
        key = (agent, dataset)
        if key not in by_key or mtime > by_key[key][1]:
            by_key[key] = (path, mtime)

    for (agent, dataset), (path, _) in sorted(by_key.items()):
        queues = parse_last_test_episode(path)
        if queues is None:
            continue
        g = gini(queues)
        mean_q = sum(queues) / len(queues)
        max_q = max(queues)
        min_q = min(queues)
        results.append({
            "agent":      agent,
            "dataset":    dataset,
            "n_inters":   len(queues),
            "gini_queue": g,
            "mean_queue": mean_q,
            "max_queue":  max_q,
            "min_queue":  min_q,
        })

    if not results:
        print("ERROR: no BRF logs found or no ep 199 data")
        return

    print(f"{'agent':<14} {'dataset':<22} {'n':>3} {'Gini':>7} {'mean_q':>8} {'min':>6} {'max':>6}")
    print("-" * 75)
    for r in results:
        print(f"{r['agent']:<14} {r['dataset']:<22} {r['n_inters']:>3} "
              f"{r['gini_queue']:>7.4f} {r['mean_queue']:>8.3f} "
              f"{r['min_queue']:>6.3f} {r['max_queue']:>6.3f}")

    with open(OUTPUT_CSV, "w") as f:
        f.write("agent,dataset,n_inters,gini_queue,mean_queue,min_queue,max_queue\n")
        for r in results:
            f.write(f"{r['agent']},{r['dataset']},{r['n_inters']},"
                    f"{r['gini_queue']:.4f},{r['mean_queue']:.4f},"
                    f"{r['min_queue']:.4f},{r['max_queue']:.4f}\n")
    print(f"\nGuardado: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
