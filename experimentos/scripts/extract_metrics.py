"""
Extrae métricas finales de los logs de LibSignal por algoritmo x dataset.
Genera results/metrics_summary.csv con ATT, AQL, Delay, Throughput.

Para algoritmos entrenables (PressLight, MPLight, CoLight, FRAP, AXLight):
    Lee el último episodio TEST de los DTL logs (formato:
    agent  TRAIN|TEST  episode  ATT  loss  reward  queue  delay  throughput)

Para baselines no entrenables (Fixed Time, Max Pressure):
    Lee la línea "Final Travel Time is..." de los BRF logs (formato:
    Final Travel Time is X, mean rewards: Y, queue: Z, delay: W, throughput: T)

Ejecutar desde: experimentos/
    conda run -n tfm python scripts/extract_metrics.py
"""
import os
import re
import csv
import glob

LOG_DIR = "LibSignal/data/output_data"
OUT_CSV = "results/metrics_summary.csv"

# Formato DTL: agent  TRAIN|TEST  episode  ATT  loss  reward  queue  delay  throughput
DTL_RE = re.compile(
    r"^(\S+)\s+(TRAIN|TEST)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([-\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)"
)

# Formato BRF (baselines): "Final Travel Time is X, mean rewards: Y, queue: Z, delay: W, throughput: T"
BRF_FINAL_RE = re.compile(
    r"Final Travel Time is ([\d.]+),\s*mean rewards:\s*([-\d.]+),\s*queue:\s*([\d.]+),\s*delay:\s*([\d.]+),\s*throughput:\s*(\d+)"
)


def find_dtl_logs():
    pattern = os.path.join(LOG_DIR, "**", "*_DTL.log")
    return sorted(glob.glob(pattern, recursive=True))


def find_baseline_brf_logs():
    """BRF logs para fixedtime y maxpressure en exp01."""
    logs = []
    for agent in ("fixedtime", "maxpressure"):
        pattern = os.path.join(LOG_DIR, "tsc", f"cityflow_{agent}", "*", "exp01", "logger", "*_BRF.log")
        logs.extend(glob.glob(pattern))
    return sorted(logs)


def parse_brf_baseline(log_path):
    """Lee la primera línea 'Final Travel Time is...' del BRF de un baseline."""
    with open(log_path, "r") as f:
        for line in f:
            m = BRF_FINAL_RE.search(line)
            if m:
                return {
                    "ATT":        float(m.group(1)),
                    "reward":     float(m.group(2)),
                    "AQL":        float(m.group(3)),
                    "delay":      float(m.group(4)),
                    "throughput": int(m.group(5)),
                }
    return None


def parse_last_test(log_path):
    """
    Lee el DTL log y devuelve las métricas del último episodio TEST.
    Devuelve None si el log no tiene ningún episodio TEST.
    """
    last_test = None
    with open(log_path, "r") as f:
        for line in f:
            m = DTL_RE.match(line.strip())
            if m and m.group(2) == "TEST":
                last_test = {
                    "agent":      m.group(1),
                    "episode":    int(m.group(3)),
                    "ATT":        float(m.group(4)),
                    "loss":       float(m.group(5)),
                    "reward":     float(m.group(6)),
                    "AQL":        float(m.group(7)),
                    "delay":      float(m.group(8)),
                    "throughput": int(m.group(9)),
                }
    return last_test


def parse_path(log_path):
    """Extrae algoritmo y dataset del path del log."""
    parts = log_path.replace("\\", "/").split("/")
    agent_seg = next((p for p in parts if p.startswith("cityflow_")), None)
    if agent_seg:
        agent = agent_seg.replace("cityflow_", "")
        idx = parts.index(agent_seg)
        dataset = parts[idx + 1] if idx + 1 < len(parts) else "unknown"
    else:
        agent = "unknown"
        dataset = "unknown"
    return agent, dataset


def main():
    rows = []

    # 1. Algoritmos entrenables: leer último TEST de los DTL logs
    by_key = {}
    for log_path in find_dtl_logs():
        agent_path, dataset = parse_path(log_path)
        if "exp01" not in log_path:
            continue
        mtime = os.path.getmtime(log_path)
        key = (agent_path, dataset)
        if key not in by_key or mtime > by_key[key][1]:
            by_key[key] = (log_path, mtime)

    for (agent, dataset), (log_path, _) in sorted(by_key.items()):
        result = parse_last_test(log_path)
        if result is None:
            print(f"  [SIN TEST]   {log_path}")
            continue
        ep = result["episode"]
        ep_warn = f" (ep {ep}, INCOMPLETO)" if ep < 199 else f" (ep {ep})"
        rows.append({
            "algoritmo":  agent,
            "dataset":    dataset,
            "episodio":   ep,
            "ATT":        round(result["ATT"], 2),
            "AQL":        round(result["AQL"], 4),
            "delay":      round(result["delay"], 4),
            "throughput": result["throughput"],
            "log_file":   os.path.basename(log_path),
        })
        print(
            f"  {agent:12s} | {dataset:20s} | "
            f"ATT={result['ATT']:.2f}  AQL={result['AQL']:.4f}  "
            f"delay={result['delay']:.4f}  tput={result['throughput']}{ep_warn}"
        )

    # 2. Baselines no entrenables: leer BRF
    by_key_brf = {}
    for log_path in find_baseline_brf_logs():
        agent_path, dataset = parse_path(log_path)
        mtime = os.path.getmtime(log_path)
        key = (agent_path, dataset)
        if key not in by_key_brf or mtime > by_key_brf[key][1]:
            by_key_brf[key] = (log_path, mtime)

    for (agent, dataset), (log_path, _) in sorted(by_key_brf.items()):
        result = parse_brf_baseline(log_path)
        if result is None:
            continue
        rows.append({
            "algoritmo":  agent,
            "dataset":    dataset,
            "episodio":   0,
            "ATT":        round(result["ATT"], 2),
            "AQL":        round(result["AQL"], 4),
            "delay":      round(result["delay"], 4),
            "throughput": result["throughput"],
            "log_file":   os.path.basename(log_path),
        })
        print(
            f"  {agent:12s} | {dataset:20s} | "
            f"ATT={result['ATT']:.2f}  AQL={result['AQL']:.4f}  "
            f"delay={result['delay']:.4f}  tput={result['throughput']} (baseline)"
        )

    if not rows:
        print("No se extrajeron métricas.")
        return

    rows.sort(key=lambda r: (r["dataset"], r["algoritmo"]))

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fieldnames = ["algoritmo", "dataset", "episodio", "ATT", "AQL", "delay", "throughput", "log_file"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV generado: {OUT_CSV} ({len(rows)} filas)")


if __name__ == "__main__":
    main()
