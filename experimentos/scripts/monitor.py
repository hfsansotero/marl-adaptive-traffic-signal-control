"""
Monitor de entrenamiento en tiempo real para LibSignal.
Lee los DTL logs activos y muestra progreso estilo Stable-Baselines3.

Uso (desde cualquier directorio):
    python scripts/monitor.py                        # busca logs bajo ./data/output_data
    python scripts/monitor.py --logdir /ruta/custom  # directorio de logs explícito
    python scripts/monitor.py --interval 5           # refresco cada 5s
    python scripts/monitor.py --once                 # imprimir una vez y salir
    python scripts/monitor.py --all                  # mostrar todos los logs, no solo activos

"Activo"  = log modificado hace menos de ACTIVE_THRESHOLD_S segundos.
"Pausado" = log modificado hace más de ACTIVE_THRESHOLD_S pero menos de STALE_THRESHOLD_S.
"Terminado" = log sin modificaciones recientes y episodio 199 visto en TEST.
"""
import os
import re
import glob
import time
import argparse
from datetime import datetime, timedelta

DTL_RE = re.compile(
    r"^(\S+)\s+(TRAIN|TEST)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([-\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)"
)

# Un episodio de FRAP/AXLight puede tardar >7 min entre escrituras de log.
# Usamos 900s (15 min) como umbral de "activo": si el log fue actualizado
# hace menos de 15 min, el proceso casi seguro sigue corriendo.
ACTIVE_THRESHOLD_S = 900
# Logs más viejos que esto se consideran terminados / abandonados.
STALE_THRESHOLD_S  = 3600 * 6  # 6 horas


def log_status(path):
    """Devuelve 'active', 'paused' o 'stale' según la antigüedad del log."""
    age = time.time() - os.path.getmtime(path)
    if age < ACTIVE_THRESHOLD_S:
        return "active"
    if age < STALE_THRESHOLD_S:
        return "paused"
    return "stale"


def find_logs(logdir, only_active=True):
    pattern = os.path.join(logdir, "**", "*_DTL.log")
    all_logs = glob.glob(pattern, recursive=True)
    if only_active:
        logs = [p for p in all_logs if log_status(p) in ("active", "paused")]
    else:
        logs = all_logs
    return sorted(logs, key=os.path.getmtime, reverse=True)


def parse_dtl(path):
    rows = {"TRAIN": [], "TEST": []}
    # timestamps: (mode, ep, wall_time) parsed from log filename + line order
    timestamps = {"TRAIN": [], "TEST": []}
    try:
        with open(path) as f:
            for line in f:
                m = DTL_RE.match(line.strip())
                if m:
                    agent, mode, ep, att, loss, reward, queue, delay, throughput = m.groups()
                    rows[mode].append({
                        "agent":      agent,
                        "ep":         int(ep),
                        "ATT":        float(att),
                        "loss":       float(loss),
                        "reward":     float(reward),
                        "queue":      float(queue),
                        "delay":      float(delay),
                        "throughput": int(throughput),
                    })
    except (FileNotFoundError, PermissionError):
        pass
    return rows


def extract_meta(log_path):
    parts = log_path.replace("\\", "/").split("/")
    agent_seg = next((p for p in parts if p.startswith("cityflow_")), None)
    if agent_seg:
        agent = agent_seg.replace("cityflow_", "")
        idx = parts.index(agent_seg)
        dataset = parts[idx + 1] if idx + 1 < len(parts) else "?"
    else:
        agent = "?"
        dataset = "?"
    # extract start datetime from log filename: YYYY_MM_DD-HH_MM_SS_DTL.log
    fname = os.path.basename(log_path)
    m = re.match(r"(\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2})_DTL", fname)
    start_dt = None
    if m:
        try:
            start_dt = datetime.strptime(m.group(1), "%Y_%m_%d-%H_%M_%S")
        except ValueError:
            pass
    return agent, dataset, start_dt


def eta_str(ep_done, total, secs_per_ep):
    if ep_done == 0 or secs_per_ep is None:
        return "calculando..."
    remaining = (total - ep_done) * secs_per_ep
    return str(timedelta(seconds=int(remaining)))


def progress_bar(n, total, width=30):
    filled = int(width * n / max(total, 1))
    return f"[{'█' * filled}{'░' * (width - filled)}] {n}/{total}"


def age_str(seconds):
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds/60)}m {int(seconds%60)}s"
    return f"{int(seconds/3600)}h {int((seconds%3600)/60)}m"



def render_log(log_path, total_ep=200):
    agent, dataset, start_dt = extract_meta(log_path)
    rows = parse_dtl(log_path)
    train = rows["TRAIN"]
    test  = rows["TEST"]
    status = log_status(log_path)
    mtime  = os.path.getmtime(log_path)
    age    = time.time() - mtime
    now    = time.time()

    # Estado
    if status == "active":
        status_icon = "● ACTIVO  "
    elif status == "paused":
        status_icon = "◌ PAUSADO "
    else:
        status_icon = "○ ANTIGUO "

    # Inicio del entrenamiento
    if start_dt:
        started_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        elapsed_s = now - start_dt.timestamp()
        started_str += f"  (hace {age_str(elapsed_s)})"
    else:
        started_str = "desconocido"

    if not train:
        return (
            f"  ┌─ {agent.upper():12s} │ {dataset}\n"
            f"  │  Inicio: {started_str}  [{status_icon}]\n"
            f"  └─ esperando primer episodio...\n"
        )

    last_train = train[-1]
    last_test  = test[-1]  if test  else None
    prev_test  = test[-2]  if len(test)  >= 2 else None
    prev_train = train[-2] if len(train) >= 2 else None
    ep_done    = last_train["ep"] + 1

    # Best TEST ATT ever seen
    best_test_att = min((r["ATT"] for r in test), default=None)

    # Episode duration estimate from mtime / episodes written
    # TRAIN episodes: ep_done total; TEST episodes: len(test) total
    # Total episodes written = ep_done (TRAIN) + len(test) (TEST)
    # Each TRAIN ep is followed by a TEST ep → pairs
    total_writes = ep_done + len(test)
    elapsed_total = now - (start_dt.timestamp() if start_dt else mtime - age)
    # Rough: split elapsed by writes
    secs_per_write = elapsed_total / max(total_writes, 1)
    # One "cycle" = 1 TRAIN + 1 TEST write
    secs_per_cycle = secs_per_write * 2
    secs_per_ep = secs_per_cycle

    lines = []
    lines.append(f"  ┌─ {agent.upper():12s} │ {dataset}  [{status_icon}│ última act.: {age_str(age)} atrás]")
    lines.append(f"  │  Inicio: {started_str}")
    lines.append(
        f"  │  {progress_bar(ep_done, total_ep)}  "
        f"ETA {eta_str(ep_done, total_ep, secs_per_ep)}"
    )

    train_col = f"TRAIN (ep {last_train['ep']})"
    test_col  = f"TEST  (ep {last_test['ep']})" if last_test else "TEST  --"
    lines.append(f"  │")
    lines.append(f"  │  {'Métrica':<14} {train_col:>24}   {test_col:>24}")
    lines.append(f"  │  {'─' * 65}")

    metrics = [
        ("ATT (s)",     "ATT",        "{:.1f}",  True),
        ("Loss",        "loss",       "{:.3f}",  True),
        ("Mean reward", "reward",     "{:.2f}",  False),
        ("Queue",       "queue",      "{:.2f}",  True),
        ("Delay",       "delay",      "{:.4f}",  True),
        ("Throughput",  "throughput", "{:d}",    False),
    ]

    for label, key, fmt, down_is_good in metrics:
        t_val = fmt.format(last_train[key])
        if last_test:
            v_val = fmt.format(last_test[key])
            if prev_test:
                diff = last_test[key] - prev_test[key]
                if abs(diff) < 1e-6:
                    arrow = "  "
                elif (diff < 0) == down_is_good:
                    arrow = "✓"
                else:
                    arrow = "✗"
            else:
                arrow = "  "

            if key == "ATT":
                is_best   = last_test["ATT"] == best_test_att
                best_flag = " ★" if is_best else "  "
                prev_v    = fmt.format(prev_test[key]) if prev_test else "—"
                prev_t    = fmt.format(prev_train[key]) if prev_train else "—"
                lines.append(f"  │  {label:<14} {prev_t:>12} → {t_val:<10}   {prev_v:>10} → {v_val:<6} {arrow}{best_flag}")
            elif key == "reward":
                prev_t = fmt.format(prev_train[key]) if prev_train else "—"
                prev_v = fmt.format(prev_test[key])  if prev_test  else "—"
                lines.append(f"  │  {label:<14} {prev_t:>12} → {t_val:<10}   {prev_v:>10} → {v_val:<8} {arrow}")
            else:
                lines.append(f"  │  {label:<14} {t_val:>24}   {v_val:>22} {arrow}")
        else:
            if key in ("ATT", "reward"):
                prev_t = fmt.format(prev_train[key]) if prev_train else "—"
                lines.append(f"  │  {label:<14} {prev_t:>12} → {t_val:<10}   {'—':>24}")
            else:
                lines.append(f"  │  {label:<14} {t_val:>24}   {'—':>24}")

    lines.append(f"  └─ última escritura: {datetime.fromtimestamp(mtime).strftime('%H:%M:%S')}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Monitor de entrenamiento LibSignal")
    parser.add_argument("--logdir",   default="data/output_data",
                        help="Directorio raíz donde buscar logs (default: data/output_data)")
    parser.add_argument("--interval", type=int, default=10,
                        help="Segundos entre refrescos (default: 10)")
    parser.add_argument("--total",    type=int, default=200,
                        help="Total de episodios configurados (default: 200)")
    parser.add_argument("--once",     action="store_true",
                        help="Imprimir una vez y salir")
    parser.add_argument("--all",      action="store_true",
                        help="Mostrar todos los logs, no solo los activos/pausados")
    parser.add_argument("--agent",    default=None,
                        help="Filtrar por nombre de agente, ej: --agent frap")
    args = parser.parse_args()

    while True:
        logs = find_logs(args.logdir, only_active=not args.all)
        if args.agent:
            logs = [p for p in logs if args.agent.lower() in extract_meta(p)[0].lower()]
        active_count  = sum(1 for p in logs if log_status(p) == "active")
        paused_count  = sum(1 for p in logs if log_status(p) == "paused")

        print("\033[2J\033[H", end="")
        print(f"{'═' * 70}")
        print(f"  LibSignal Training Monitor  │  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(
            f"  ● Activos: {active_count}  "
            f"◌ Pausados (entre episodios, <15 min): {paused_count}  "
            f"│  umbral activo: {ACTIVE_THRESHOLD_S}s"
        )
        print(f"{'═' * 70}")

        if not logs:
            print(f"\n  Sin entrenamientos en: {os.path.abspath(args.logdir)}")
            print(f"\n  Si hay un experimento corriendo pero no aparece, prueba:")
            print(f"    python scripts/monitor.py --all   (muestra todos los logs)")
            print(f"\n  O lanza un experimento con:")
            print(f"    python run.py -a frap -w cityflow -n cityflow4x4 --ngpu 0")
        else:
            for log in logs[:8]:
                print(render_log(log, total_ep=args.total))
                print()

        print(f"{'─' * 70}")
        hint = "--all  " if not args.all else "--all activo"
        print(f"  Refresco cada {args.interval}s  │  Ctrl+C para salir  │  {hint} para ver todos los logs")

        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Monitor detenido.")
